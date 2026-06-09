"""
Helper functions for printing model information after POI likelihood fits.

These functions are used across backends (zmodel, hfmodel, roomodel) to provide
a consistent interface for printing PDFs and floating parameter values.
"""

import sys


def print_model_info(fit_model, summary, backend_name, state=None):
    """Print model information after POI likelihood fit.
    
    Parameters
    ----------
    fit_model : object
        The fit model object (backend-specific).
    summary : dict
        The summary dictionary containing fit results.
    backend_name : str
        The name of the backend ('zmodel', 'hfmodel', 'roomodel').
    state : object, optional
        Backend-specific state object (used for some backends).
    """
    if backend_name == "zmodel":
        _print_zmodel_info(fit_model, summary)
    elif backend_name == "hfmodel":
        _print_hfmodel_info(fit_model, summary)
    elif backend_name == "roomodel":
        _print_roomodel_info(fit_model, summary, state)
    else:
        print(f"Unknown backend: {backend_name}", file=sys.stderr)


def _print_zmodel_info(fit_model, summary):
    """Print zfit model PDFs and floating parameters."""
    print("\n=== Model Information (zfit) ===")
    
    # Print PDFs
    if hasattr(fit_model, "model") and fit_model.model is not None:
        model = fit_model.model
        print(f"\nModel: {model}")
        
        # Print PDFs from the model
        if hasattr(model, "pdfs"):
            pdfs = model.pdfs
            if pdfs:
                print(f"\nPDFs ({len(pdfs)}):")
                for i, pdf in enumerate(pdfs):
                    print(f"  [{i}] {pdf}")
        
        # Print all model parameters
        if hasattr(model, "parameters"):
            params = model.parameters
            if params:
                print(f"\nModel Parameters ({len(params)}):")
                for param in params:
                    print(f"  {param}")
    
    # Print floating parameter values after fit
    if summary and "fit_params" in summary:
        fit_params = summary.get("fit_params", {})
        fit_param_unc = summary.get("fit_param_unc", {})
        
        if fit_params:
            print(f"\nFloating Parameters After POI Fit ({len(fit_params)}):")
            for param_name in sorted(fit_params.keys()):
                value = fit_params[param_name]
                unc = fit_param_unc.get(param_name)
                
                if unc is not None:
                    print(f"  {param_name:30s} = {value:+.6e} ± {unc:.6e}")
                else:
                    print(f"  {param_name:30s} = {value:+.6e}")


def _print_hfmodel_info(fit_model, summary):
    """Print pyhf model PDFs and floating parameters."""
    print("\n=== Model Information (pyhf) ===")
    
    # Print workspace info
    if hasattr(fit_model, "workspace"):
        workspace = fit_model.workspace
        print(f"\nWorkspace: {fit_model.model_name if hasattr(fit_model, 'model_name') else 'unknown'}")
        
        # Print channels
        if hasattr(fit_model, "channels"):
            channels = fit_model.channels
            print(f"\nChannels ({len(channels)}):")
            for channel_name in channels:
                print(f"  - {channel_name}")
        
        # Print processes
        if hasattr(workspace, "list_of_processes"):
            procs = workspace.list_of_processes
            print(f"\nProcesses ({len(procs)}):")
            for proc in procs:
                print(f"  - {proc}")
    
    # Print floating parameter values after fit
    fit_params = summary.get("fit_params", {}) if summary else {}
    fit_param_unc = summary.get("fit_param_unc", {}) if summary else {}
    
    if fit_params:
        print(f"\nFloating Parameters After POI Fit ({len(fit_params)}):")
        for param_name in sorted(fit_params.keys()):
            value = fit_params[param_name]
            unc = fit_param_unc.get(param_name)
            
            if unc is not None:
                print(f"  {param_name:30s} = {value:+.6e} ± {unc:.6e}")
            else:
                print(f"  {param_name:30s} = {value:+.6e}")


def _print_roomodel_info(fit_model, summary, state=None):
    """Print RooFit model PDFs and floating parameters."""
    print("\n=== Model Information (RooFit) ===")
    
    # Print model info
    print(f"\nModel: {fit_model.model_name if hasattr(fit_model, 'model_name') else 'unknown'}")
    
    if hasattr(fit_model, "metadata") and fit_model.metadata:
        metadata = fit_model.metadata
        
        # Print channels
        if "channels" in metadata:
            channels = metadata["channels"]
            print(f"\nChannels ({len(channels)}):")
            for channel_name in channels:
                print(f"  - {channel_name}")
        
        # Print processes
        if "processes" in metadata:
            processes = metadata["processes"]
            print(f"\nProcesses ({len(processes)}):")
            for proc in processes:
                print(f"  - {proc}")
    
    # Print RooFit workspace contents if available
    if state is not None and hasattr(state, "workspace"):
        workspace = state.workspace
        print(f"\nRooFit Workspace PDFs:")
        # Try to list PDFs in workspace
        try:
            pdf_set = workspace.allPdfs()
            if pdf_set:
                for obj in pdf_set:
                    print(f"  {obj.GetName()}")
        except Exception:
            pass
    
    # Print floating parameter values after fit
    fit_params = summary.get("fit_params", {}) if summary else {}
    fit_param_unc = summary.get("fit_param_unc", {}) if summary else {}
    
    if fit_params:
        print(f"\nFloating Parameters After POI Fit ({len(fit_params)}):")
        for param_name in sorted(fit_params.keys()):
            value = fit_params[param_name]
            unc = fit_param_unc.get(param_name)
            
            if unc is not None:
                print(f"  {param_name:30s} = {value:+.6e} ± {unc:.6e}")
            else:
                print(f"  {param_name:30s} = {value:+.6e}")
