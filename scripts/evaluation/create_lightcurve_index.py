#!/usr/bin/env python
"""
Create an HTML index file for easy navigation through lightcurve visualizations.

This script scans a directory for lightcurve HTML files and creates an index page
with thumbnails, navigation, and easy access to individual lightcurves.

Usage:
    python create_lightcurve_index.py /path/to/lightcurve/directory
    python scripts/evaluation/create_lightcurve_index.py /sps/lsst/users/rbonnetguerrini/ML4transients/saved/lc/highconfv4
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import os
from pathlib import Path
import re
from datetime import datetime
import sys

# Add the src path to import ML4transients modules
script_dir = Path(__file__).parent
src_path = script_dir.parent.parent / "src"
sys.path.insert(0, str(src_path))


def extract_dia_object_id(filename):
    """Extract diaObjectId from lightcurve filename."""
    match = re.search(r'lightcurve_(\d+)\.html', filename)
    return match.group(1) if match else None


def get_udeep_dataset_loader():
    """Initialize DatasetLoader for the full UDEEP dataset where SNN inference was performed."""
    try:
        from ML4transients.data_access.dataset_loader import DatasetLoader
        udeep_path = Path("/sps/lsst/groups/transients/HSC/fouchez/raphael/data/UDEEP_coadd")
        return DatasetLoader(udeep_path)
    except ImportError as e:
        print(f"Could not import DatasetLoader: {e}")
        print("Make sure you're running with the correct conda environment")
        return None


def get_lightcurve_info_from_udeep(dia_object_id, dataset_loader):
    """
    Get lightcurve length, SNN inference results, and average flux from the UDEEP dataset.
    
    Parameters
    ----------
    dia_object_id : int
        The diaObjectId to look up
    dataset_loader : DatasetLoader
        The UDEEP dataset loader
        
    Returns
    -------
    dict
        Dictionary with 'length', 'snn_info', 'avg_flux'
    """
    try:
        import pandas as pd
        import numpy as np
        
        # Get lightcurve data
        lc = dataset_loader.lightcurves.get_lightcurve(int(dia_object_id))
        
        if lc is not None:
            lc_length = len(lc)
            # Calculate average flux (assuming 'psfFlux' column exists)
            if 'psfFlux' in lc.columns:
                avg_flux = lc['psfFlux'].mean()
                avg_flux_str = f"{avg_flux:.2e}" if not pd.isna(avg_flux) else "N/A"
            else:
                avg_flux_str = "N/A"
        else:
            lc_length = "N/A"
            avg_flux_str = "N/A"
        
        # Get SNN inference results - use cached version if available
        if not hasattr(dataset_loader.lightcurves, '_cached_snn_data'):
            print("Loading SNN inference data (this may take a moment)...")
            snn_data = dataset_loader.lightcurves.load_snn_inference(
                columns=["diaObjectId", "prob_class1_mean", "prob_class1_std", "pred_class"]
            )
            dataset_loader.lightcurves._cached_snn_data = snn_data
        else:
            snn_data = dataset_loader.lightcurves._cached_snn_data
        
        if snn_data is not None:
            # Convert diaObjectId to int for comparison
            if snn_data["diaObjectId"].dtype == 'object':
                snn_data_copy = snn_data.copy()
                snn_data_copy["diaObjectId"] = snn_data_copy["diaObjectId"].astype(str).astype(int)
            else:
                snn_data_copy = snn_data
            
            # Find the row for this diaObjectId
            mask = snn_data_copy["diaObjectId"] == int(dia_object_id)
            matching_rows = snn_data_copy[mask]
            
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                snn_prob = row["prob_class1_mean"]
                snn_std = row["prob_class1_std"]
                
                # Format like in the original code
                if not (pd.isna(snn_prob) or pd.isna(snn_std)):
                    snn_info = f"{snn_prob:.3f} ± {snn_std:.3f}"
                else:
                    snn_info = "N/A"
            else:
                snn_info = "N/A"
        else:
            snn_info = "N/A"
        
        return {
            'length': lc_length,
            'snn_info': snn_info,
            'avg_flux': avg_flux_str
        }
        
    except Exception as e:
        print(f"Error getting info for diaObjectId {dia_object_id}: {e}")
        return {
            'length': "N/A",
            'snn_info': "N/A",
            'avg_flux': "N/A"
        }


def create_lightcurve_index(directory_path, output_filename="index.html"):
    """
    Create an HTML index file for lightcurve visualizations.
    
    Parameters
    ----------
    directory_path : str or Path
        Path to directory containing lightcurve HTML files
    output_filename : str
        Name of the output index file
    """
    import pandas as pd
    
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return
    
    # Find all lightcurve HTML files
    lightcurve_files = list(directory_path.glob("lightcurve_*.html"))
    
    if not lightcurve_files:
        print(f"No lightcurve HTML files found in {directory_path}")
        return
    
    # Sort files by diaObjectId (numerically)
    lightcurve_files.sort(key=lambda f: int(extract_dia_object_id(f.name) or 0))
    
    # Initialize UDEEP dataset loader once for all lookups
    print("Loading UDEEP dataset for lightcurve info and SNN inference...")
    try:
        udeep_loader = get_udeep_dataset_loader()
        print("UDEEP dataset loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load UDEEP dataset: {e}")
        print("Will show 'N/A' for lightcurve length and SNN info")
        udeep_loader = None
    
    # Create index HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightcurve Index - {directory_path.name}</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            background-color: #1a1a1a;
            color: #ffffff;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: #2a2a2a;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        
        .header h1 {{
            color: #870000;
            margin: 0;
            font-size: 2.5em;
        }}
        
        .header p {{
            color: #cccccc;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .stat-item {{
            background-color: #333333;
            padding: 10px 15px;
            border-radius: 5px;
            border-left: 4px solid #870000;
        }}
        
        .controls {{
            text-align: center;
            margin-bottom: 30px;
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 10px;
        }}
        
        .controls-row {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }}
        
        .sort-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .sort-label {{
            color: #870000;
            font-weight: bold;
            font-size: 14px;
        }}
        
        .sort-select {{
            padding: 8px;
            background-color: #333;
            color: white;
            border: 2px solid #666;
            border-radius: 5px;
            font-size: 14px;
        }}
        
        .sort-select:focus {{
            outline: none;
            border-color: #870000;
        }}
        
        .search-box {{
            padding: 10px;
            font-size: 16px;
            border: 2px solid #666;
            border-radius: 5px;
            background-color: #333;
            color: white;
            width: 300px;
            margin-right: 10px;
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: #870000;
        }}
        
        .btn {{
            padding: 10px 20px;
            background-color: #870000;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
            transition: background-color 0.3s;
        }}
        
        .btn:hover {{
            background-color: #a50000;
        }}
        
        .lightcurve-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .lightcurve-card {{
            background-color: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.3s, box-shadow 0.3s;
            border: 1px solid #444;
        }}
        
        .lightcurve-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.4);
            border-color: #870000;
        }}
        
        .lightcurve-card h3 {{
            color: #870000;
            margin: 0 0 10px 0;
            font-size: 1.3em;
            word-break: break-all;
        }}
        
        .lightcurve-info {{
            color: #cccccc;
            margin-bottom: 15px;
            font-size: 0.9em;
        }}
        
        .lightcurve-actions {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .action-btn {{
            padding: 8px 15px;
            background-color: #4a4a4a;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 14px;
            transition: background-color 0.3s;
            border: 1px solid #666;
            flex: 1;
            text-align: center;
            min-width: 80px;
        }}
        
        .action-btn:hover {{
            background-color: #870000;
            text-decoration: none;
            color: white;
        }}
        
        .action-btn.primary {{
            background-color: #870000;
            border-color: #870000;
        }}
        
        .action-btn.primary:hover {{
            background-color: #a50000;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background-color: #2a2a2a;
            border-radius: 10px;
            color: #888;
        }}
        
        .hidden {{
            display: none;
        }}
        
        .navigation {{
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid #444;
            z-index: 1000;
        }}
        
        .navigation h4 {{
            color: #870000;
            margin: 0 0 10px 0;
            font-size: 14px;
        }}
        
        .nav-buttons {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .nav-btn {{
            padding: 5px 10px;
            background-color: #4a4a4a;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
            transition: background-color 0.3s;
        }}
        
        .nav-btn:hover {{
            background-color: #870000;
        }}
        
        @media (max-width: 768px) {{
            .lightcurve-grid {{
                grid-template-columns: 1fr;
            }}
            
            .navigation {{
                position: relative;
                width: 100%;
                margin-bottom: 20px;
            }}
            
            .stats {{
                flex-direction: column;
                align-items: center;
            }}
        }}
    </style>
</head>
<body>
    <div class="navigation">
        <h4>Quick Navigation</h4>
        <div class="nav-buttons">
            <button class="nav-btn" onclick="scrollToTop()">Top</button>
            <button class="nav-btn" onclick="scrollToBottom()">Bottom</button>
            <button class="nav-btn" onclick="toggleView()">Toggle View</button>
        </div>
    </div>

    <div class="header">
        <h1>Lightcurve Index</h1>
        <p>Dataset: <strong>{directory_path.name}</strong></p>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div class="stats">
            <div class="stat-item">
                <strong>Total Lightcurves:</strong> {len(lightcurve_files)}
            </div>
            <div class="stat-item">
                <strong>Directory:</strong> {directory_path.name}
            </div>
        </div>
    </div>

    <div class="controls">
        <div class="controls-row">
            <input type="text" class="search-box" id="searchBox" placeholder="Search by diaObjectId..." onkeyup="filterLightcurves()">
            <button class="btn" onclick="clearSearch()">Clear</button>
            <button class="btn" onclick="openAllVisible()">Open All Visible</button>
        </div>
        <div class="controls-row">
            <div class="sort-controls">
                <span class="sort-label">Sort by:</span>
                <select class="sort-select" id="sortSelect" onchange="sortLightcurves()">
                    <option value="id">diaObjectId</option>
                    <option value="length">LC Length</option>
                    <option value="flux">Avg Flux</option>
                    <option value="snn">SNN Probability</option>
                </select>
                <button class="btn" onclick="toggleSortOrder()" id="sortOrderBtn">↑ Asc</button>
            </div>
        </div>
    </div>

    <div class="lightcurve-grid" id="lightcurveGrid">
"""

    # Add each lightcurve card
    for i, file_path in enumerate(lightcurve_files):
        dia_object_id = extract_dia_object_id(file_path.name)
        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        
        # Get lightcurve info and SNN inference from UDEEP dataset
        if udeep_loader is not None:
            lc_info = get_lightcurve_info_from_udeep(dia_object_id, udeep_loader)
            lc_length = lc_info['length']
            snn_info = lc_info['snn_info']
            avg_flux = lc_info['avg_flux']
        else:
            lc_length = "N/A"
            snn_info = "N/A"
            avg_flux = "N/A"
        
        # Prepare sortable values
        length_val = lc_length if isinstance(lc_length, (int, float)) else -1
        flux_val = avg_flux if avg_flux != "N/A" else -1
        snn_val = -1
        
        # Extract numeric value from SNN info for sorting
        if snn_info != "N/A" and "±" in snn_info:
            try:
                snn_val = float(snn_info.split("±")[0].strip())
            except:
                snn_val = -1
        
        # Convert scientific notation flux to float for sorting
        if isinstance(avg_flux, str) and avg_flux != "N/A":
            try:
                flux_val = float(avg_flux)
            except:
                flux_val = -1
        
        html_content += f"""
        <div class="lightcurve-card" data-dia-id="{dia_object_id}" 
             data-length="{length_val}" 
             data-flux="{flux_val}" 
             data-snn="{snn_val}">
            <h3>diaObjectId: {dia_object_id}</h3>
            <div class="lightcurve-info">
                <div><strong>Modified:</strong> {mod_time}</div>
                <div><strong>LC Length:</strong> {lc_length}</div>
                <div><strong>SNN Prob:</strong> {snn_info}</div>
                <div><strong>Avg Flux:</strong> {avg_flux}</div>
            </div>
            <div class="lightcurve-actions">
                <a href="{file_path.name}" class="action-btn primary" target="_blank">Open Lightcurve</a>
                <button class="action-btn" onclick="copyId('{dia_object_id}')">Copy ID</button>
            </div>
        </div>
"""

    # Close HTML and add JavaScript
    html_content += f"""
    </div>

    <div class="footer">
        <p>Lightcurve Index for {directory_path.name} | {len(lightcurve_files)} files | Generated by ML4transients</p>
        <p>Navigate between lightcurves using the search box or browse the grid above</p>
    </div>

    <script>
        function filterLightcurves() {{
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            const cards = document.querySelectorAll('.lightcurve-card');
            let visibleCount = 0;
            
            cards.forEach(card => {{
                const diaId = card.getAttribute('data-dia-id').toLowerCase();
                if (diaId.includes(searchTerm)) {{
                    card.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    card.classList.add('hidden');
                }}
            }});
            
            // Update visible count in navigation
            console.log(`Showing ${{visibleCount}} of {len(lightcurve_files)} lightcurves`);
        }}
        
        function clearSearch() {{
            document.getElementById('searchBox').value = '';
            filterLightcurves();
        }}
        
        function openAllVisible() {{
            const visibleCards = document.querySelectorAll('.lightcurve-card:not(.hidden)');
            if (visibleCards.length > 10) {{
                if (!confirm(`This will open ${{visibleCards.length}} lightcurves. Continue?`)) {{
                    return;
                }}
            }}
            
            visibleCards.forEach(card => {{
                const link = card.querySelector('.action-btn.primary');
                if (link) {{
                    window.open(link.href, '_blank');
                }}
            }});
        }}
        
        function copyId(diaId) {{
            navigator.clipboard.writeText(diaId).then(() => {{
                console.log('Copied diaObjectId: ' + diaId);
                // Could add a temporary visual indicator here
            }});
        }}
        
        // Sorting functionality
        let currentSortOrder = 'asc';
        
        function sortLightcurves() {{
            const sortBy = document.getElementById('sortSelect').value;
            const grid = document.getElementById('lightcurveGrid');
            const cards = Array.from(grid.children);
            
            cards.sort((a, b) => {{
                let valueA, valueB;
                
                switch (sortBy) {{
                    case 'id':
                        valueA = parseInt(a.getAttribute('data-dia-id'));
                        valueB = parseInt(b.getAttribute('data-dia-id'));
                        break;
                    case 'length':
                        valueA = parseFloat(a.getAttribute('data-length'));
                        valueB = parseFloat(b.getAttribute('data-length'));
                        break;
                    case 'flux':
                        valueA = parseFloat(a.getAttribute('data-flux'));
                        valueB = parseFloat(b.getAttribute('data-flux'));
                        break;
                    case 'snn':
                        valueA = parseFloat(a.getAttribute('data-snn'));
                        valueB = parseFloat(b.getAttribute('data-snn'));
                        break;
                    default:
                        return 0;
                }}
                
                // Handle N/A values (represented as -1)
                if (valueA === -1 && valueB === -1) return 0;
                if (valueA === -1) return 1;  // Put N/A at the end
                if (valueB === -1) return -1;
                
                // Sort logic
                if (currentSortOrder === 'asc') {{
                    return valueA - valueB;
                }} else {{
                    return valueB - valueA;
                }}
            }});
            
            // Clear the grid and re-append sorted cards
            grid.innerHTML = '';
            cards.forEach(card => grid.appendChild(card));
        }}
        
        function toggleSortOrder() {{
            currentSortOrder = currentSortOrder === 'asc' ? 'desc' : 'asc';
            const button = document.getElementById('sortOrderBtn');
            button.textContent = currentSortOrder === 'asc' ? '↑ Asc' : '↓ Desc';
            sortLightcurves();
        }}
        
        function scrollToTop() {{
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}
        
        function scrollToBottom() {{
            window.scrollTo({{ top: document.body.scrollHeight, behavior: 'smooth' }});
        }}
        
        function toggleView() {{
            const grid = document.getElementById('lightcurveGrid');
            if (grid.style.gridTemplateColumns === '1fr') {{
                grid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(350px, 1fr))';
            }} else {{
                grid.style.gridTemplateColumns = '1fr';
            }}
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.ctrlKey || e.metaKey) {{
                switch(e.key) {{
                    case 'f':
                        e.preventDefault();
                        document.getElementById('searchBox').focus();
                        break;
                    case 'k':
                        e.preventDefault();
                        clearSearch();
                        break;
                }}
            }}
        }});
        
        // Auto-focus search box on page load
        window.addEventListener('load', function() {{
            document.getElementById('searchBox').focus();
        }});
    </script>
</body>
</html>"""

    # Write the index file
    output_path = directory_path / output_filename
    output_path.write_text(html_content, encoding='utf-8')
    
    print(f"Created lightcurve index: {output_path}")
    print(f"Found {len(lightcurve_files)} lightcurve files")
    print(f"\nOpen the index file in your browser:")
    print(f"  file://{output_path.absolute()}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HTML index for lightcurve visualizations")
    parser.add_argument("directory", help="Directory containing lightcurve HTML files")
    parser.add_argument("--output", "-o", default="index.html", 
                       help="Output filename (default: index.html)")
    
    args = parser.parse_args()
    
    create_lightcurve_index(args.directory, args.output)