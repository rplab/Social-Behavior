# -*- coding: utf-8 -*-
"""
Author:   Raghuveer Parthasarathy
Created on Fri May 23 18:53:58 2025
Last modified Fri May 23 18:53:58 2025 -- Raghuveer Parthasarathy

Description
-----------

Code to visualize fish body positions, headings, and other information
stored in datasets[], etc.
The code is almost completely from Claude 4 Sonnet 
(May 23- 24, 2025; see “Code notes May 2025.docx”)

Example usage:
    
j=0
vis = visualize_fish_data(all_position_data[j], 
                          datasets[j]["heading_angle"], 
                          datasets[j]["head_head_vec_px"], 
                          CSVcolumns["body_column_x_start"], 
                          CSVcolumns["body_column_y_start"], 
                          CSVcolumns["body_Ncolumns"], 
                          additional_info = datasets[j]["relative_orientation"])


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

class FishVisualizer:
    def __init__(self, position_data, heading_angles, dh_vec_px, 
                 body_column_x_start, body_column_y_start, body_Ncolumns,
                 additional_info=None):
        """
        Interactive fish position visualizer
        
        Parameters:
        -----------
        position_data : numpy.ndarray
            Shape (Nframes, M, Nfish) - fish body positions
        heading_angles : numpy.ndarray
            Shape (Nframes, Nfish) - heading angles in radians
        dh_vec_px : numpy.ndarray
            Shape (Nframes, Nfish-1) - head-to-head vectors
        body_column_x_start : int
            Starting column index for x positions
        body_column_y_start : int
            Starting column index for y positions
        body_Ncolumns : int
            Number of body points per fish
        additional_info : numpy.ndarray, optional
            Shape (Nframes, Nfish) - additional information to display
        """
        self.position_data = position_data
        self.heading_angles = heading_angles
        self.dh_vec_px = dh_vec_px
        self.body_column_x_start = body_column_x_start
        self.body_column_y_start = body_column_y_start
        self.body_Ncolumns = body_Ncolumns
        self.additional_info = additional_info
        
        self.Nframes, self.M, self.Nfish = position_data.shape
        self.current_frame = 0
        self.show_text = 0  # 0: no text, 1: angles, 2: additional_info (if available)
        self.autoscale = True  # New: toggle for autoscaling
        
        # Generate distinct colors for each fish
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.Nfish))
        
        # Set up the plot with focus capability
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect('equal')
        
        # Set window title (if supported by backend)
        try:
            self.fig.canvas.manager.set_window_title('Fish Visualizer - Click here first, then use keys')
        except:
            # Fallback if window title setting fails
            pass
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initialize plot elements
        self.head_circles = []
        self.body_points = []
        self.body_lines = []
        self.heading_arrows = []
        self.dh_arrows = []
        self.angle_texts = []
        
        # Plot initial frame
        self.plot_frame()
        
    def get_fish_positions(self, frame, fish_idx):
        """Get x,y positions for a specific fish in a specific frame"""
        x_cols = slice(self.body_column_x_start, 
                      self.body_column_x_start + self.body_Ncolumns)
        y_cols = slice(self.body_column_y_start, 
                      self.body_column_y_start + self.body_Ncolumns)
        
        x_pos = self.position_data[frame, x_cols, fish_idx]
        y_pos = self.position_data[frame, y_cols, fish_idx]
        
        return x_pos, y_pos
    
    def clear_plot_elements(self):
        """Clear all plot elements"""
        for elements in [self.head_circles, self.body_points, self.body_lines, 
                        self.heading_arrows, self.dh_arrows, self.angle_texts]:
            for element in elements:
                element.remove()
            elements.clear()
    
    def plot_frame(self):
        """Plot fish positions for current frame"""
        self.clear_plot_elements()
        
        # Get current frame data
        frame = self.current_frame
        
        # Plot each fish
        for fish_idx in range(self.Nfish):
            color = self.colors[fish_idx]
            
            # Get fish positions
            x_pos, y_pos = self.get_fish_positions(frame, fish_idx)
            
            # Skip if positions contain NaN
            if np.any(np.isnan(x_pos)) or np.any(np.isnan(y_pos)):
                continue
            
            # Plot head position as circle
            head_circle = self.ax.plot(x_pos[0], y_pos[0], 'o', 
                                     color=color, markersize=8, 
                                     label=f'Fish {fish_idx}')[0]
            self.head_circles.append(head_circle)
            
            # Plot body positions as x's
            if len(x_pos) > 1:
                body_x = self.ax.plot(x_pos[1:], y_pos[1:], 'x', 
                                    color=color, markersize=6)[0]
                self.body_points.append(body_x)
            
            # Connect positions with lines
            body_line = self.ax.plot(x_pos, y_pos, '-', 
                                   color=color, alpha=0.7)[0]
            self.body_lines.append(body_line)
            
            # Plot heading angle arrow
            heading_angle = self.heading_angles[frame, fish_idx]
            if not np.isnan(heading_angle):
                arrow_length = 20  # Adjust as needed
                dx = arrow_length * np.cos(heading_angle)
                dy = arrow_length * np.sin(heading_angle)
                
                arrow = FancyArrowPatch((x_pos[0], y_pos[0]),
                                      (x_pos[0] + dx, y_pos[0] + dy),
                                      arrowstyle='->', mutation_scale=15,
                                      color=color, linewidth=2)
                self.ax.add_patch(arrow)
                self.heading_arrows.append(arrow)
                
                # Add text display if enabled
                if self.show_text == 1:  # Show heading angles
                    angle_deg = np.degrees(heading_angle)
                    text = self.ax.text(x_pos[0] + dx/2, y_pos[0] + dy/2,
                                      f'{angle_deg:.1f}°',
                                      color=color, fontsize=12,  # 50% larger (was 8)
                                      ha='center', va='center',
                                      bbox=dict(boxstyle='round,pad=0.2',
                                              facecolor='white', alpha=0.7))
                    self.angle_texts.append(text)
                elif self.show_text == 2 and self.additional_info is not None:  # Show additional info
                    add_info = self.additional_info[frame, fish_idx]
                    if not np.isnan(add_info):
                        text = self.ax.text(x_pos[0] + dx/2, y_pos[0] + dy/2,
                                          f'{add_info:.2f}',
                                          color=color, fontsize=12,  # 50% larger (was 8)
                                          ha='center', va='center',
                                          bbox=dict(boxstyle='round,pad=0.2',
                                                  facecolor='lightblue', alpha=0.7))
                        self.angle_texts.append(text)
        
        # Plot head-to-head vectors
        for i in range(self.Nfish - 1):  # Fixed: should be Nfish-1, not len(dh_vec_px[frame])
            if not np.isnan(self.dh_vec_px[frame, i]):
                # Get head positions of consecutive fish
                x_pos_i, y_pos_i = self.get_fish_positions(frame, i)
                x_pos_j, y_pos_j = self.get_fish_positions(frame, i + 1)
                
                if (not np.any(np.isnan([x_pos_i[0], y_pos_i[0], 
                                       x_pos_j[0], y_pos_j[0]]))):
                    dh_arrow = FancyArrowPatch((x_pos_i[0], y_pos_i[0]),
                                             (x_pos_j[0], y_pos_j[0]),
                                             arrowstyle='-|>', 
                                             mutation_scale=20,
                                             color='gray', alpha=0.6, linewidth=3)
                    self.ax.add_patch(dh_arrow)
                    self.dh_arrows.append(dh_arrow)
        
        # Update title and adjust view
        self.ax.set_title(f'Frame {frame} / {self.Nframes - 1}')
        
        # Handle autoscaling vs fixed limits
        if self.autoscale:
            self.ax.autoscale()
        # If not autoscaling, keep current axis limits (don't change them)
        
        # Refresh plot
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse clicks to ensure focus"""
        # This helps ensure the figure has focus for keyboard events
        self.fig.canvas.draw()
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        # print(f"Key pressed: {event.key}")  # Debug print
        
        if event.key == 'u':  # Forward 1 frame (was 'h')
            self.current_frame = min(self.current_frame + 1, self.Nframes - 1)
        elif event.key == 'i':  # Forward 10 frames (was 'j')
            self.current_frame = min(self.current_frame + 10, self.Nframes - 1)
        elif event.key == 'o':  # Forward 100 frames (was 'k')
            self.current_frame = min(self.current_frame + 100, self.Nframes - 1)
        elif event.key == 'p':  # Forward 1000 frames (was 'l')
            self.current_frame = min(self.current_frame + 1000, self.Nframes - 1)
        elif event.key == 't':  # Back 1 frame (was 'f')
            self.current_frame = max(self.current_frame - 1, 0)
        elif event.key == 'r':  # Back 10 frames (was 'd')
            self.current_frame = max(self.current_frame - 10, 0)
        elif event.key == 'e':  # Back 100 frames (was 's')
            self.current_frame = max(self.current_frame - 100, 0)
        elif event.key == 'w':  # Back 1000 frames (was 'a')
            self.current_frame = max(self.current_frame - 1000, 0)
        elif event.key == 'v':  # Toggle text display
            if self.additional_info is not None:
                # Cycle through: no text -> angles -> additional_info -> no text
                self.show_text = (self.show_text + 1) % 3
                text_modes = ['OFF', 'Heading Angles', 'Additional Info']
                print(f"Text display: {text_modes[self.show_text]}")
            else:
                # Toggle between no text and angles only
                self.show_text = 1 if self.show_text == 0 else 0
                print(f"Text display: {'Heading Angles' if self.show_text else 'OFF'}")
        elif event.key == 'z':  # Toggle autoscaling
            self.autoscale = not self.autoscale
            print(f"Autoscaling: {'ON' if self.autoscale else 'OFF'}")
        else:
            print(f"Unrecognized key: {event.key}")
            return  # Don't update plot for unrecognized keys
        
        # print(f"Frame: {self.current_frame}")
        # Replot with new frame
        self.plot_frame()
    
    def go_to_frame(self, frame_num):
        """Manually go to a specific frame (alternative to keyboard)"""
        self.current_frame = np.clip(frame_num, 0, self.Nframes - 1)
        self.plot_frame()
        print(f"Went to frame: {self.current_frame}")
    
    def next_frame(self, step=1):
        """Go forward by step frames"""
        self.current_frame = min(self.current_frame + step, self.Nframes - 1)
        self.plot_frame()
        print(f"Frame: {self.current_frame}")
    
    def prev_frame(self, step=1):
        """Go backward by step frames"""
        self.current_frame = max(self.current_frame - step, 0)
        self.plot_frame()
        print(f"Frame: {self.current_frame}")
    
    def toggle_angles(self):
        """Toggle angle display"""
        self.show_angles = not self.show_angles
        self.plot_frame()
        print(f"Angle display: {'ON' if self.show_angles else 'OFF'}")
    
    def show(self):
        """Display the interactive plot"""
        print("Interactive Fish Visualizer")
        print("=" * 50)
        print("IMPORTANT: Click on the plot window first to give it focus!")
        print("")
        print("Keyboard Controls:")
        print("  u: +1 frame    t: -1 frame")
        print("  i: +10 frames  r: -10 frames") 
        print("  o: +100 frames e: -100 frames")
        print("  p: +1000 frames w: -1000 frames")
        print("  v: toggle angle display")
        print("  z: toggle autoscaling (on/off)")
        print("")
        print("Alternative programmatic controls:")
        print("  vis.next_frame(n)    # Go forward n frames")
        print("  vis.prev_frame(n)    # Go backward n frames") 
        print("  vis.go_to_frame(n)   # Go to specific frame")
        print("  vis.toggle_angles()  # Toggle angle display")
        print("=" * 50)
        
        # Try to make the figure focused
        plt.figure(self.fig.number)
        self.fig.canvas.draw()
        
        # Force focus (backend dependent)
        try:
            self.fig.canvas.manager.window.wm_attributes('-topmost', 1)
            self.fig.canvas.manager.window.wm_attributes('-topmost', 0)
        except:
            pass
            
        plt.show()

def visualize_fish_data(position_data, heading_angles, dh_vec_px,
                       body_column_x_start, body_column_y_start, body_Ncolumns, 
                       additional_info = None):
    """
    Create and show interactive fish visualization
    
    Parameters:
    -----------
    position_data : numpy.ndarray
        Shape (Nframes, M, Nfish) - fish body positions
    heading_angles : numpy.ndarray
        Shape (Nframes, Nfish) - heading angles in radians
    dh_vec_px : numpy.ndarray
        Shape (Nframes, Nfish-1) - head-to-head vectors
    body_column_x_start : int
        Starting column index for x positions
    body_column_y_start : int
        Starting column index for y positions
    body_Ncolumns : int
        Number of body points per fish
    additional_info : numpy array of the same shape as heading_angles; 
        additional_info can be, for example, relative orientation of fish.
        
    Returns:
    --------
    FishVisualizer : The visualizer object for programmatic control
    """
    visualizer = FishVisualizer(position_data, heading_angles, dh_vec_px,
                               body_column_x_start, body_column_y_start, 
                               body_Ncolumns, additional_info = additional_info)
    visualizer.show()
    return visualizer

# Example usage:
if __name__ == "__main__":
    # Create sample data for demonstration
    Nframes = 1000
    Nfish = 5
    body_Ncolumns = 10
    M = 20  # Total columns in position_data
    
    # Sample position data
    position_data = np.random.randn(Nframes, M, Nfish) * 50 + 100
    
    # Sample heading angles
    heading_angles = np.random.uniform(0, 2*np.pi, (Nframes, Nfish))
    
    # Sample head-to-head vectors
    dh_vec_px = np.random.randn(Nframes, Nfish-1) * 10
    
    # Column indices
    body_column_x_start = 0
    body_column_y_start = 10
    
    # Create visualization and return visualizer for programmatic control
    vis = visualize_fish_data(position_data, heading_angles, dh_vec_px,
                             body_column_x_start, body_column_y_start, body_Ncolumns)
    
    # Example of programmatic control if keyboard doesn't work:
    # vis.next_frame(10)     # Go forward 10 frames
    # vis.prev_frame(5)      # Go back 5 frames  
    # vis.go_to_frame(100)   # Go to frame 100
    # vis.toggle_angles()    # Toggle angle display