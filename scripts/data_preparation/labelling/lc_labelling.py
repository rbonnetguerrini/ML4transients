#!/usr/bin/env python
"""
Lightcurve Labelling Software

Interactive tool for labelling lightcurves as real (1) or bogus (0).
Displays lightcurves with cutout images organized by filter.

Usage:
    python lc_labelling.py --data-path /path/to/data --object-ids 12345 67890 --output-dir /path/to/labels
    python lc_labelling.py --data-path /path/to/data --object-file object_ids.txt --output-dir /path/to/labels
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, HoverTool, Div, Spacer
from bokeh.io import curdoc

# Import ML4transients data loaders
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))
from ML4transients.data_access.dataset_loader import DatasetLoader

# Band colors and order
BAND_COLORS = {
    'u': '#56b4e9', 'g': '#009e73', 'r': '#e69f00',
    'i': '#cc79a7', 'z': '#d55e00', 'y': '#f0e442',
}
BAND_ORDER = ['u', 'g', 'r', 'i', 'z', 'y']


class LightcurveLabellingApp:
    """Interactive labelling application for lightcurves."""
    
    def __init__(self, dataset_loader: DatasetLoader, object_ids: List[int], output_dir: Path):
        self.dataset_loader = dataset_loader
        self.object_ids = object_ids
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.labels_file = self.output_dir / "labels.json"
        self.labels = self._load_existing_labels()
        
        self.current_index = 0
        self.current_data = None
    
    def _load_existing_labels(self) -> Dict:
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_labels(self):
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
    
    def get_current_object_id(self) -> Optional[int]:
        if 0 <= self.current_index < len(self.object_ids):
            return self.object_ids[self.current_index]
        return None
    
    def load_current_lightcurve(self):
        """Load data for the current lightcurve."""
        obj_id = self.get_current_object_id()
        if obj_id is None:
            return False
        
        print(f"\nLoading diaObjectId: {obj_id} ({self.current_index + 1}/{len(self.object_ids)})")
        
        try:
            # Ensure dataset discovery has happened
            self.dataset_loader._ensure_discovery()
            
            # Get lightcurve data
            lc_df = self.dataset_loader.get_lightcurve_by_object_id(obj_id)
            if lc_df is None or len(lc_df) == 0:
                print(f"No lightcurve data found for {obj_id}")
                return False
            
            # Group by visit for efficient loading
            visit_groups = {}
            for idx, row in lc_df.iterrows():
                visit = row['visit']
                # Use index as source ID if diaSourceId column doesn't exist
                src_id = row.get('diaSourceId', idx)
                band = row['band']
                
                if visit not in visit_groups:
                    visit_groups[visit] = {'ids': [], 'bands': {}}
                visit_groups[visit]['ids'].append(src_id)
                visit_groups[visit]['bands'][src_id] = band
            
            # Load all three types of cutouts
            cutouts_diff = {}
            cutouts_science = {}
            cutouts_coadd = {}
            
            for visit, info in visit_groups.items():
                if visit in self.dataset_loader._cutout_loaders:
                    loader = self.dataset_loader._cutout_loaders[visit]
                    cutouts_diff.update(loader.get_multiple_by_ids(info['ids'], cutout_type='diff'))
                    cutouts_science.update(loader.get_multiple_by_ids(info['ids'], cutout_type='science'))
                    cutouts_coadd.update(loader.get_multiple_by_ids(info['ids'], cutout_type='coadd'))
            
            self.current_data = {
                'object_id': obj_id,
                'lightcurve': lc_df,
                'cutouts_diff': cutouts_diff,
                'cutouts_science': cutouts_science,
                'cutouts_coadd': cutouts_coadd,
            }
            
            print(f"Loaded {len(lc_df)} lightcurve points, {len(cutouts_diff)} diff cutouts")
            return True
            
        except Exception as e:
            print(f"Error loading lightcurve for {obj_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_lightcurve_plot(self):
        """Create Bokeh plot for lightcurve (all bands on same plot)."""
        if self.current_data is None:
            return Div(text="<p>No data loaded</p>")
        
        lc_df = self.current_data['lightcurve']
        obj_id = self.current_data['object_id']
        
        p = figure(
            title=f"Lightcurve for diaObjectId {obj_id}",
            x_axis_label="MJD", y_axis_label="Flux",
            width=1000, height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            background_fill_color="#2F2F2F", border_fill_color="#2F2F2F"
        )
        
        # Plot by band
        for band in BAND_ORDER:
            band_data = lc_df[lc_df['band'] == band]
            if len(band_data) == 0:
                continue
            
            color = BAND_COLORS.get(band, '#888888')
            
            # Get source IDs (use index if column doesn't exist)
            if 'diaSourceId' in band_data.columns:
                src_ids = band_data['diaSourceId'].astype(str)
            else:
                src_ids = band_data.index.astype(str)
            
            # Get flux columns (support both psFlux and psfFlux)
            flux_col = 'psFlux' if 'psFlux' in band_data.columns else 'psfFlux'
            flux_err_col = 'psFluxErr' if 'psFluxErr' in band_data.columns else 'psfFluxErr'
            
            source = ColumnDataSource(data=dict(
                mjd=band_data['midpointMjdTai'],
                flux=band_data[flux_col],
                flux_err=band_data[flux_err_col],
                band=[band] * len(band_data),
                source_id=src_ids
            ))
            
            p.circle('mjd', 'flux', source=source, size=8, color=color, 
                    alpha=0.7, legend_label=f"Band {band}")
            
            # Error bars
            for _, row in band_data.iterrows():
                flux = row[flux_col]
                flux_err = row[flux_err_col]
                p.segment(
                    x0=row['midpointMjdTai'], y0=flux - flux_err,
                    x1=row['midpointMjdTai'], y1=flux + flux_err,
                    color=color, alpha=0.5
                )
        
        hover = HoverTool(tooltips=[
            ("MJD", "@mjd{0.0000}"), ("Flux", "@flux{0.000}"),
            ("Error", "@flux_err{0.000}"), ("Band", "@band"), ("Source ID", "@source_id")
        ])
        p.add_tools(hover)
        
        # Styling
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.background_fill_color = "#2F2F2F"
        p.legend.label_text_color = "white"
        p.title.text_color = "white"
        p.xaxis.axis_label_text_color = "white"
        p.yaxis.axis_label_text_color = "white"
        p.xaxis.major_label_text_color = "white"
        p.yaxis.major_label_text_color = "white"
        p.grid.grid_line_color = "#444"
        
        return p
    
    def create_cutout_display(self):
        """Create cutout display organized by filter (rows) and type (columns)."""
        if self.current_data is None:
            return Div(text="<p>No cutouts loaded</p>")
        
        lc_df = self.current_data['lightcurve']
        cutouts_diff = self.current_data['cutouts_diff']
        cutouts_science = self.current_data['cutouts_science']
        cutouts_coadd = self.current_data['cutouts_coadd']
        
        # Organize cutouts by band
        band_cutouts = {}
        for band in BAND_ORDER:
            band_data = lc_df[lc_df['band'] == band]
            if len(band_data) == 0:
                continue
            
            # Get source IDs (use index if column doesn't exist)
            if 'diaSourceId' in band_data.columns:
                src_ids = band_data['diaSourceId'].tolist()
            else:
                src_ids = band_data.index.tolist()
            band_cutouts[band] = {
                'diff': [cutouts_diff.get(sid) for sid in src_ids if sid in cutouts_diff],
                'science': [cutouts_science.get(sid) for sid in src_ids if sid in cutouts_science],
                'coadd': [cutouts_coadd.get(sid) for sid in src_ids if sid in cutouts_coadd]
            }
        
        header_html = """
        <div style="background-color: #2F2F2F; padding: 10px; border-radius: 5px; border: 1px solid #666;">
            <h3 style="color: #870000; margin: 0; text-align: center;">Cutout Images</h3>
            <p style="color: white; margin: 5px 0 0 0; text-align: center; font-size: 11px;">
                Each row = one filter | Columns: Coadd (template), Science (mean), Diff (mean)
            </p>
        </div>
        """
        header_div = Div(text=header_html, width=900)
        
        # Create cutout grid - flatten for gridplot with ncols
        cutout_plots = []
        for band in BAND_ORDER:
            if band not in band_cutouts:
                continue
            
            band_color = BAND_COLORS.get(band, '#888888')
            
            # Coadd (first one, same template)
            coadd_imgs = band_cutouts[band]['coadd']
            if coadd_imgs and len(coadd_imgs) > 0:
                cutout_plots.append(self._create_cutout_figure(coadd_imgs[0], f"{band} - Coadd", band_color))
            else:
                cutout_plots.append(Spacer(width=200, height=200))
            
            # Science (mean)
            science_imgs = band_cutouts[band]['science']
            if science_imgs and len(science_imgs) > 0:
                cutout_plots.append(self._create_cutout_figure(np.mean(science_imgs, axis=0), 
                                                           f"{band} - Science (mean)", band_color))
            else:
                cutout_plots.append(Spacer(width=200, height=200))
            
            # Diff (mean)
            diff_imgs = band_cutouts[band]['diff']
            if diff_imgs and len(diff_imgs) > 0:
                cutout_plots.append(self._create_cutout_figure(np.mean(diff_imgs, axis=0), 
                                                           f"{band} - Diff (mean)", band_color))
            else:
                cutout_plots.append(Spacer(width=200, height=200))
        
        if not cutout_plots:
            no_cutouts_html = """
            <div style="background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666;">
                <p style="color: #888; margin: 0; text-align: center;">No cutouts available for this object</p>
            </div>
            """
            return column(header_div, Div(text=no_cutouts_html, width=900))
        
        grid = gridplot(cutout_plots, ncols=3, width=200, height=200, 
                       toolbar_location=None, merge_tools=False)
        
        return column(header_div, grid)
    
    def _create_cutout_figure(self, img_array, title, border_color):
        """Create a Bokeh figure for a cutout image."""
        p = figure(
            title=title, width=200, height=200,
            x_range=(0, img_array.shape[1]), y_range=(0, img_array.shape[0]),
            toolbar_location=None,
            background_fill_color="#2F2F2F", border_fill_color="#2F2F2F",
            outline_line_color=border_color, outline_line_width=3
        )
        
        # Normalize image
        vmin, vmax = np.percentile(img_array, [1, 99])
        img_normalized = np.clip((img_array - vmin) / (vmax - vmin), 0, 1)
        
        p.image(image=[img_normalized], x=0, y=0, dw=img_array.shape[1], 
               dh=img_array.shape[0], palette="Greys256")
        
        # Style
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.title.text_color = "white"
        p.title.text_font_size = "10pt"
        
        return p
    
    def create_control_panel(self):
        """Create control panel with progress and info."""
        obj_id = self.get_current_object_id()
        current_label = self.labels.get(str(obj_id), None)
        
        info_html = f"""
        <div style="background-color: #2F2F2F; padding: 15px; border-radius: 5px; border: 1px solid #666;">
            <h3 style="color: #870000; margin: 0;">Lightcurve Labelling</h3>
            <p style="color: white; margin: 10px 0 0 0;">
                <strong>Progress:</strong> {self.current_index + 1} / {len(self.object_ids)}<br>
                <strong>Object ID:</strong> {obj_id}<br>
                <strong>Current Label:</strong> {current_label if current_label is not None else 'Not labelled'}<br>
                <strong>Total Labelled:</strong> {len(self.labels)} / {len(self.object_ids)}
            </p>
            <p style="color: #888; margin: 10px 0 0 0; font-size: 11px;">
                Use interactive console or call app.save_label(object_id, label) to label<br>
                0 = Bogus | 1 = Real
            </p>
        </div>
        """
        
        return Div(text=info_html, width=900)
    
    def save_label(self, object_id: int, label: int):
        """Save a label for an object."""
        self.labels[str(object_id)] = label
        self._save_labels()
        print(f"Saved label {label} for object {object_id}")
    
    def next_object(self):
        if self.current_index < len(self.object_ids) - 1:
            self.current_index += 1
            return True
        return False
    
    def previous_object(self):
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False
    
    def create_layout(self):
        """Create complete layout for the application."""
        if not self.load_current_lightcurve():
            return Div(text="<p style='color: red;'>Failed to load lightcurve data</p>")
        
        control_panel = self.create_control_panel()
        lc_plot = self.create_lightcurve_plot()
        cutout_display = self.create_cutout_display()
        
        return column(control_panel, lc_plot, cutout_display)
    
    def export_html(self, output_file: str = None):
        """Export current view to HTML file."""
        if output_file is None:
            obj_id = self.get_current_object_id()
            output_file = self.output_dir / f"label_{obj_id}.html"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        from bokeh.plotting import output_file as bokeh_output_file
        
        curdoc().theme = 'dark_minimal'
        layout = self.create_layout()
        
        bokeh_output_file(str(output_file))
        save(layout)
        
        print(f"Saved visualization to {output_file}")
        return output_file


def batch_export_html(dataset_loader: DatasetLoader, object_ids: List[int], 
                     output_dir: Path, max_objects: int = None):
    """Export HTML files for multiple objects."""
    if max_objects:
        object_ids = object_ids[:max_objects]
    
    print(f"Exporting {len(object_ids)} objects to {output_dir}")
    app = LightcurveLabellingApp(dataset_loader, object_ids, output_dir)
    
    for i, obj_id in enumerate(object_ids):
        print(f"[{i+1}/{len(object_ids)}] Exporting {obj_id}...")
        app.current_index = i
        try:
            app.export_html()
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nExport complete! Files saved to {output_dir}")


def interactive_labelling(dataset_loader: DatasetLoader, object_ids: List[int], output_dir: Path):
    """Interactive console-based labelling session."""
    app = LightcurveLabellingApp(dataset_loader, object_ids, output_dir)
    
    print("\n" + "="*60)
    print("LIGHTCURVE LABELLING - Commands: 0/1, s/skip, n/next, p/prev, v/view, q/quit")
    print("="*60)
    
    while True:
        obj_id = app.get_current_object_id()
        if obj_id is None:
            break
        
        if not app.load_current_lightcurve():
            if not app.next_object():
                break
            continue
        
        current_label = app.labels.get(str(obj_id))
        print(f"\nObject {app.current_index + 1}/{len(object_ids)}: {obj_id} [label: {current_label}]")
        
        cmd = input("Command: ").lower().strip()
        
        if cmd in ['0', 'bogus']:
            app.save_label(obj_id, 0)
            app.next_object()
        elif cmd in ['1', 'real']:
            app.save_label(obj_id, 1)
            app.next_object()
        elif cmd in ['s', 'skip']:
            app.next_object()
        elif cmd in ['n', 'next']:
            app.next_object()
        elif cmd in ['p', 'prev']:
            app.previous_object()
        elif cmd in ['v', 'view']:
            app.export_html()
        elif cmd in ['q', 'quit']:
            break
    
    print(f"\nSession complete. Labelled: {len(app.labels)}/{len(object_ids)}")


def main():
    parser = argparse.ArgumentParser(description="Lightcurve labelling tool")
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--object-file', type=str,
                       help='Path to file containing object IDs (one per line). If not provided, processes all objects.')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--export-html', action='store_true')
    parser.add_argument('--max-objects', type=int)
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_loader = DatasetLoader(args.data_path)
    
    # Load object IDs
    if args.object_file:
        with open(args.object_file, 'r') as f:
            object_ids = [int(line.strip()) for line in f if line.strip()]
    else:
        # Get all object IDs from the dataset
        # Try to load from lightcurve index directly
        import pandas as pd
        from pathlib import Path
        
        lc_path = Path(args.data_path) / 'lightcurves'
        index_file = lc_path / 'lightcurve_index.h5'
        
        if index_file.exists():
            lc_index = pd.read_hdf(index_file)
            object_ids = lc_index.index.tolist()
            print(f"Found {len(object_ids)} objects in lightcurve index")
        else:
            parser.error(f"No lightcurve index found at {index_file} and no --object-file provided")
    
    if args.max_objects:
        object_ids = object_ids[:args.max_objects]
    
    print(f"Processing {len(object_ids)} objects")
    
    if args.export_html:
        batch_export_html(dataset_loader, object_ids, Path(args.output_dir), args.max_objects)
    else:
        interactive_labelling(dataset_loader, object_ids, Path(args.output_dir))


if __name__ == "__main__":
    main()
