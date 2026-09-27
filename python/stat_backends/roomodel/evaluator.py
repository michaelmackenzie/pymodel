"""C++ evaluation loop for the roomodel likelihood (declared once through cling).

All model quantities are RooFit objects (yields, morphing formulas, pdfs, bin integrals);
this helper only loops over bins/events and processes in C++ so that one NLL evaluation is
a single call from python.  It computes the ``inference.model`` convention exactly::

    binned / counting:  sum_p Y_p T_p - sum_i n_i log mu_i,   mu_i = sum_p Y_p f_pi
    unbinned:           sum_p Y_p - sum_j w_j log(sum_p Y_p d_p(x_j))

where f_pi is the fraction of process p in bin i, d_p the normalised density and T_p = 1
(T_p = 0 for an empty template), i.e. the extended term is the full yield even when the
bin-centre fractions of a parametric pdf do not sum to 1 (Combine's CachingAddNLL).  A process
has one or more *states* (several for an envelope, selected by its RooCategory); a state
provides its bin fractions as constants, as RooAbsReal functions (template morphing, bin
integrals), or by evaluating a pdf at the bin centre times the bin width.
"""

from modelspec import rootinput as R

_CODE = r"""
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "RooAbsPdf.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooCategory.h"
#include "RooRealVar.h"

namespace pymodel_roo {

struct State {
  std::vector<double> cfrac;        // constant bin fractions
  std::vector<RooAbsReal*> ffrac;   // bin fraction functions
  RooAbsPdf* pdf = nullptr;         // normalised density over the observable
  double total = 1.0;               // nu_p(total) / yield: 1, or 0 for an empty template
};

struct Proc {
  RooAbsReal* yield = nullptr;
  RooCategory* cat = nullptr;
  std::vector<State> states;
  const State& state() const { return states.at(cat ? cat->getCurrentIndex() : 0); }
};

class Channel {
public:
  Channel(RooRealVar* obs, const std::vector<double>& edges) : obs_(obs), edges_(edges) {
    if (obs_) nset_.add(*obs_);
    work_.assign(nbins(), 0.0);
    tmp_.assign(nbins(), 0.0);
  }
  int nbins() const { return int(edges_.size()) - 1; }
  int nprocs() const { return int(procs_.size()); }
  int add_proc(RooAbsReal* yield, RooCategory* cat) {
    Proc p; p.yield = yield; p.cat = cat; procs_.push_back(p); return nprocs() - 1;
  }
  // A state's bin fractions come from ``cfrac`` if not empty, else from ``ffrac`` if not
  // empty, else from ``pdf`` at the bin centres; unbinned densities come from ``pdf`` if set,
  // else from the bin fractions divided by the bin width.
  void add_state(int ip, const std::vector<double>& cfrac, const std::vector<RooAbsReal*>& ffrac, RooAbsPdf* pdf,
                 double total) {
    State s; s.cfrac = cfrac; s.ffrac = ffrac; s.pdf = pdf; s.total = total; procs_.at(ip).states.push_back(s);
  }

  // bin fractions of the current state of process ip
  void fractions(int ip, double* out) {
    const State& s = procs_.at(ip).state();
    const int nb = nbins();
    if (!s.cfrac.empty()) {
      for (int i = 0; i < nb; ++i) out[i] = s.cfrac[i];
    } else if (!s.ffrac.empty()) {
      for (int i = 0; i < nb; ++i) out[i] = s.ffrac[i]->getVal();
    } else {
      for (int i = 0; i < nb; ++i) {
        obs_->setVal(0.5 * (edges_[i] + edges_[i + 1]));
        out[i] = s.pdf->getVal(nset_) * (edges_[i + 1] - edges_[i]);
      }
    }
  }
  double proc_yield(int ip) { return procs_.at(ip).yield->getVal(); }
  void expected(int ip, double* out) {
    fractions(ip, out);
    const double y = proc_yield(ip);
    for (int i = 0; i < nbins(); ++i) out[i] *= y;
  }

  double binned_nll(const double* n) {
    const int nb = nbins();
    std::fill(work_.begin(), work_.end(), 0.0);
    double ytot = 0.0;
    for (int ip = 0; ip < nprocs(); ++ip) {
      const double y = proc_yield(ip);
      ytot += y * procs_.at(ip).state().total;
      fractions(ip, tmp_.data());
      for (int i = 0; i < nb; ++i) work_[i] += y * tmp_[i];
    }
    double total = ytot;
    for (int i = 0; i < nb; ++i) {
      const double mu = work_[i];
      if (!(mu > 0)) {
        if (n[i] > 0 || std::isnan(mu)) return std::numeric_limits<double>::infinity();
        continue;  // mu <= 0 with no data: no log term
      }
      if (n[i] != 0) total -= n[i] * std::log(mu);
    }
    return total;
  }

  // normalised density of the current state of process ip at x (obs already set to x)
  double density(const State& s, double x) {
    if (s.pdf) return s.pdf->getVal(nset_);
    const int nb = nbins();
    if (x < edges_[0] || x > edges_[nb]) return 0.0;
    int k = int(std::upper_bound(edges_.begin(), edges_.end(), x) - edges_.begin()) - 1;
    if (k >= nb) k = nb - 1;
    const double f = s.cfrac.empty() ? s.ffrac[k]->getVal() : s.cfrac[k];
    return f / (edges_[k + 1] - edges_[k]);
  }

  double unbinned_nll(const double* x, const double* w, int ne) {
    dens_.assign(ne, 0.0);
    double ytot = 0.0;
    for (int ip = 0; ip < nprocs(); ++ip) {
      const double y = proc_yield(ip);
      ytot += y;
      const State& s = procs_.at(ip).state();
      for (int j = 0; j < ne; ++j) {
        obs_->setVal(x[j]);
        dens_[j] += y * density(s, x[j]);
      }
    }
    double total = ytot;
    for (int j = 0; j < ne; ++j) {
      const double wj = w ? w[j] : 1.0;
      if (wj == 0) continue;
      if (!(dens_[j] > 0)) return std::numeric_limits<double>::infinity();
      total -= wj * std::log(dens_[j]);
    }
    return total;
  }

private:
  RooRealVar* obs_;
  RooArgSet nset_;
  std::vector<double> edges_;
  std::vector<Proc> procs_;
  std::vector<double> work_, tmp_, dens_;
};

}  // namespace pymodel_roo
"""

_DECLARED = False


def cpp():
    """The ``pymodel_roo`` C++ namespace (declared on first use)."""
    global _DECLARED
    ROOT = R.root()
    if not _DECLARED:
        if not ROOT.gInterpreter.Declare(_CODE):
            raise RuntimeError("roomodel: could not declare the C++ evaluation helper")
        _DECLARED = True
    return ROOT.pymodel_roo
