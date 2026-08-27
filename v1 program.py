import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons, Button
from matplotlib.collections import LineCollection
import fastf1


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7

CACHE_DIR = 'f1_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)



def get_input(prompt_text, default=None):
    """Captures terminal input while allowing 'M' to return to the Main Menu."""
    val = input(prompt_text).strip()
    if val.upper() == 'M':
        return 'MAIN_MENU'
    if not val and default is not None:
        return default
    return val



class AeroDynamicsEngine:
    def __init__(self, df, mass_kg=798, wheelbase_m=3.6, track_width_m=1.6, cg_height_m=0.3,
                 steering_ratio=14.0, k_front=150000, k_rear=180000, cla=3.8, cda=1.2,
                 static_rh_front=35.0, static_rh_rear=75.0):
        self.df = df.copy().reset_index(drop=True)
        self.mass = mass_kg
        self.wb = wheelbase_m
        self.tw = track_width_m
        self.cg_h = cg_height_m
        self.steering_ratio = steering_ratio
        self.k_front = k_front 
        self.k_rear = k_rear    
        self.cla = cla
        self.cda = cda
        self.static_rh_f = static_rh_front
        self.static_rh_r = static_rh_rear
        self.rho = 1.225

    def process_all(self):
        self._calculate_kinematics()
        self._calculate_aero_and_ride_height()
        self._calculate_cop_and_yaw()
        self._calculate_platform_and_balance()
        self._extract_coastdown_cda()
        return self.df

    def _calculate_kinematics(self):
        self.df['Speed_ms'] = np.maximum(self.df['Speed'] / 3.6, 0.1)
        
        if 'Time' in self.df.columns and pd.api.types.is_timedelta64_dtype(self.df['Time']):
            dt = self.df['Time'].dt.total_seconds().diff().fillna(0.02).values
        else:
            dt = np.full(len(self.df), 0.02)
        dt = np.where(dt <= 0, 0.02, dt)
        self.df['dt'] = dt
        
        self.df['ax_g'] = (np.gradient(self.df['Speed_ms']) / dt) / 9.81
        
        if 'X' in self.df.columns and 'Y' in self.df.columns:
            x = self.df['X'].rolling(window=5, min_periods=1, center=True).mean().values
            y = self.df['Y'].rolling(window=5, min_periods=1, center=True).mean().values
            
            dx, dy = np.gradient(x), np.gradient(y)
            ddx, ddy = np.gradient(dx), np.gradient(dy)
            
            denom = np.maximum((dx**2 + dy**2)**1.5, 1e-6)
            curvature = np.clip(np.abs(dx * ddy - dy * ddx) / denom, 0.0, 0.1)
            
            self.df['Radius_m'] = np.where(curvature > 1e-4, 1.0 / curvature, 1000.0)
            self.df['Kinematic_Steering_deg'] = np.degrees(np.arctan(self.wb / self.df['Radius_m'])) * self.steering_ratio
            
            cross_prod = np.sign(dx * ddy - dy * ddx)
            self.df['ay_g'] = (self.df['Speed_ms']**2 / (self.df['Radius_m'] * 9.81)) * cross_prod
        else:
            self.df['Kinematic_Steering_deg'] = 5.0
            self.df['ay_g'] = self.df.get('G_Lat', 0.0)
            cross_prod = np.sign(self.df['ay_g'])

        if 'SteeringWheelAngle' in self.df.columns and not self.df['SteeringWheelAngle'].isnull().all():
            self.df['Steering_Angle_deg'] = self.df['SteeringWheelAngle']
        elif 'Steer_Angle' in self.df.columns and not self.df['Steer_Angle'].isnull().all():
            self.df['Steering_Angle_deg'] = self.df['Steer_Angle']
        else:
            ku = 1.8 + (self.df['Speed'] / 300.0) * 0.9
            slip_steer_deg = ku * np.abs(self.df['ay_g'])
            self.df['Steering_Angle_deg'] = (self.df['Kinematic_Steering_deg'] + slip_steer_deg) * cross_prod

    def _calculate_aero_and_ride_height(self):
        q = 0.5 * self.rho * (self.df['Speed_ms']**2)
        self.df['Downforce_N'] = q * self.cla
        self.df['Drag_N'] = q * self.cda
        
        f_aero_front = self.df['Downforce_N'] * 0.45
        f_aero_rear = self.df['Downforce_N'] * 0.55
        
        self.df['RideHeight_Front_mm'] = np.maximum(self.static_rh_f - ((f_aero_front / (2 * self.k_front)) * 1000), 1.0)
        self.df['RideHeight_Rear_mm'] = np.maximum(self.static_rh_r - ((f_aero_rear / (2 * self.k_rear)) * 1000), 2.0)

    def _calculate_cop_and_yaw(self):
        rh_delta = self.df['RideHeight_Rear_mm'] - self.df['RideHeight_Front_mm']
        self.df['CoP_Front_Pct'] = np.clip(45.0 + (rh_delta * 0.12) + (self.df['Speed'] * 0.015), 38.0, 52.0)

        beta_rad = np.arctan2(self.df['ay_g'] * 9.81 * (self.wb * 0.5), np.maximum(self.df['Speed_ms']**2, 1.0))
        self.df['Yaw_Angle_deg'] = np.degrees(beta_rad)

    def _calculate_platform_and_balance(self):
        self.df['Understeer_Delta_deg'] = np.abs(self.df['Steering_Angle_deg']) - self.df['Kinematic_Steering_deg']
        
        weight_transfer_long = (self.df['ax_g'] * self.mass * 9.81 * self.cg_h) / self.wb
        weight_transfer_lat = (self.df['ay_g'] * self.mass * 9.81 * self.cg_h) / self.tw
        
        self.df['Pitch_Angle_deg'] = np.degrees(np.arctan(weight_transfer_long / (self.k_front + self.k_rear)))
        self.df['Roll_Angle_deg'] = np.degrees(np.arctan(weight_transfer_lat / (self.k_front + self.k_rear)))

    def _extract_coastdown_cda(self):
        throttle = self.df['Throttle'].values if 'Throttle' in self.df.columns else np.zeros(len(self.df))
        brake = self.df['Brake'].values if 'Brake' in self.df.columns else np.zeros(len(self.df))
        
        coast_mask = (throttle < 2.0) & (brake < 1.0) & (self.df['Speed'] > 100.0) & (self.df['ax_g'] < 0)
        
        f_drag_extracted = -self.df['ax_g'] * self.mass * 9.81 - (0.015 * self.mass * 9.81)
        q = 0.5 * self.rho * (self.df['Speed_ms']**2)
        cda_extracted = np.where(coast_mask & (q > 0), f_drag_extracted / q, np.nan)
        self.df['Extracted_CdA'] = np.clip(cda_extracted, 0.5, 2.5)


    def __init__(self, filepath=None, freestream_velocity_ms=50.0):
        self.filepath = filepath
        self.v_inf = freestream_velocity_ms
        self.q_inf = 0.5 * 1.225 * (self.v_inf ** 2)

    def load_and_process(self, seed=42, df_bias=1.0):
        if self.filepath and os.path.exists(self.filepath):
            df_raw = pd.read_csv(self.filepath)
            df_raw.columns = df_raw.columns.str.strip().str.lower()
            return self._parse_user_csv(df_raw)
        else:
            return self._generate_synthetic_star_ccm_mesh(seed=seed, df_bias=df_bias)

    def _parse_user_csv(self, df):
        rename_map = {'x': 'X', 'y': 'Y', 'z': 'Z', 'pressure': 'Pressure', 'cp': 'Cp', 'area': 'Area'}
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        if 'Cp' not in df.columns and 'Pressure' in df.columns:
            df['Cp'] = df['Pressure'] / self.q_inf
        if 'Area' not in df.columns:
            df['Area'] = 0.001
        df['dFz_N'] = -df['Cp'] * self.q_inf * df['Area']
        return df

    def _generate_synthetic_star_ccm_mesh(self, seed=42, df_bias=1.0):
        np.random.seed(seed)
        n_points = 2500
        x = np.random.uniform(-1.0, 3.5, n_points)
        y = np.random.uniform(-0.8, 0.8, n_points)
        z = np.random.uniform(0.01, 0.6, n_points)
        
        cp = np.zeros(n_points)
        for i in range(n_points):
            if x[i] < 0:
                cp[i] = np.random.normal(-1.8 * df_bias, 0.3)
            elif 0 <= x[i] <= 2.2 and z[i] < 0.15:
                cp[i] = np.random.normal(-2.2 * df_bias, 0.4)
            elif x[i] > 2.8 and z[i] > 0.4:
                cp[i] = np.random.normal(-1.4 * df_bias, 0.2)
            else:
                cp[i] = np.random.normal(0.2, 0.3)
                
        area = np.random.uniform(0.0005, 0.002, n_points)
        df_mesh = pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'Cp': cp, 'Area': area})
        df_mesh['Pressure_Pa'] = df_mesh['Cp'] * self.q_inf
        df_mesh['dFz_N'] = -df_mesh['Pressure_Pa'] * df_mesh['Area']
        return df_mesh



def draw_telemetry_overlay(ax, runs_list, selected_driver_idx):
    colors = ['crimson', 'royalblue', 'forestgreen']
    for i, item in enumerate(runs_list):
        label, df = item['label'], item['data']
        lw = 2.0 if i == selected_driver_idx else 1.0
        alpha = 1.0 if i == selected_driver_idx else 0.4
        ax.plot(df['Distance'], df['Speed'], color=colors[i % len(colors)], linewidth=lw, alpha=alpha, label=label)
    ax.set_title("Multi-Driver Speed Overlay", pad=4)
    ax.set_ylabel("Speed [km/h]")
    ax.set_xlabel("Distance [m]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=7)

def draw_cop_understeer(ax, driver_item):
    df = driver_item['data']
    ax2 = ax.twinx()
    l1 = ax.plot(df['Distance'], df['CoP_Front_Pct'], color='crimson', label='CoP Front %')
    l2 = ax2.plot(df['Distance'], df['Understeer_Delta_deg'], color='navy', linestyle='--', label='Understeer Delta [°]')
    ax.set_title(f"Aero Balance (CoP) & Understeer Gradient ({driver_item['label']})", pad=4)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Front CoP [%]", color='crimson')
    ax2.set_ylabel("Steering Delta [°]", color='navy')
    ax.grid(True, linestyle=':', alpha=0.6)

def draw_yaw_sensitivity(ax, driver_item):
    df = driver_item['data']
    sc = ax.scatter(df['Yaw_Angle_deg'], df['Downforce_N'] / 1000.0, c=df['Speed'], cmap='magma', s=12, alpha=0.7)
    ax.set_title(f"Yaw Angle Sensitivity vs Downforce ({driver_item['label']})", pad=4)
    ax.set_xlabel("Yaw Angle [°]")
    ax.set_ylabel("Downforce [kN]")
    ax.grid(True, linestyle=':', alpha=0.6)

def draw_coastdown_cda(ax, driver_item):
    df = driver_item['data']
    valid_cda = df.dropna(subset=['Extracted_CdA'])
    if not valid_cda.empty:
        ax.scatter(valid_cda['Speed'], valid_cda['Extracted_CdA'], color='darkorange', s=15, alpha=0.8, label='Extracted CdA')
        ax.axhline(df['cda'].iloc[0] if 'cda' in df.columns else 1.2, color='black', linestyle='--', label='Theoretical CdA')
    else:
        ax.text(0.5, 0.5, "No Coasting Phase Found (0% Throttle/Brake)", ha='center', va='center')
    ax.set_title(f"Coast-Down Drag ($C_d A$) Extraction ({driver_item['label']})", pad=4)
    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("$C_d A$ [$m^2$]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=7)

def draw_top_speed_clipping(ax, driver_item):
    df = driver_item['data']
    high_speed = df[df['Speed'] > 220.0]
    if not high_speed.empty:
        ax.scatter(high_speed['Speed'], high_speed['ax_g'], c=high_speed['Throttle'], cmap='cool', s=15)
        ax.axhline(0, color='red', linestyle='--', lw=0.8)
    else:
        ax.plot(df['Speed'], df['ax_g'], color='teal')
    ax.set_title(f"Top Speed Clipping Analysis ({driver_item['label']})", pad=4)
    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("Longitudinal Accel [g]")
    ax.grid(True, linestyle=':', alpha=0.6)

def draw_aero_banded_gg(ax, driver_item):
    df = driver_item['data']
    bands = [
        (df['Speed'] < 120, 'royalblue', 'Low Speed (<120)'),
        ((df['Speed'] >= 120) & (df['Speed'] < 220), 'gold', 'Mid Speed (120-220)'),
        (df['Speed'] >= 220, 'crimson', 'High Speed (>220)')
    ]
    for mask, color, label in bands:
        sub = df[mask]
        ax.scatter(sub['ay_g'], sub['ax_g'], color=color, s=10, alpha=0.6, label=label)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_title(f"Aero-Banded Traction Circle ({driver_item['label']})", pad=4)
    ax.set_xlabel("Lat Accel [g]")
    ax.set_ylabel("Long Accel [g]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=7)

def draw_corner_profiling(ax, driver_item):
    df = driver_item['data']
    low_speed = df[df['Speed'] < 120]['ay_g'].abs().max()
    mid_speed = df[(df['Speed'] >= 120) & (df['Speed'] < 220)]['ay_g'].abs().max()
    high_speed = df[df['Speed'] >= 220]['ay_g'].abs().max()
    
    categories = ['Low Speed\n(Mech Grip)', 'Mid Speed\n(Transition)', 'High Speed\n(Aero Grip)']
    values = [np.nan_to_num(low_speed), np.nan_to_num(mid_speed), np.nan_to_num(high_speed)]
    
    ax.bar(categories, values, color=['mediumseagreen', 'orange', 'crimson'])
    ax.set_title(f"Corner Grip Segmentation ({driver_item['label']})", pad=4)
    ax.set_ylabel("Peak Lateral Accel [g]")
    ax.grid(True, linestyle=':', alpha=0.5)

def draw_porpoising_fft(ax, driver_item):
    df = driver_item['data']
    straight_mask = (df['Speed'] > 220.0) & (df['ay_g'].abs() < 0.5)
    straight_data = df[straight_mask]
    
    if len(straight_data) > 30:
        rh_signal = straight_data['RideHeight_Front_mm'].values - straight_data['RideHeight_Front_mm'].mean()
        dt = straight_data['dt'].mean() if 'dt' in straight_data.columns else 0.02
        fft_vals = np.abs(np.fft.rfft(rh_signal))
        freqs = np.fft.rfftfreq(len(rh_signal), d=dt)
        
        ax.plot(freqs, fft_vals, color='purple', lw=1.5)
        ax.set_xlim(1, 20)
        ax.axvspan(5, 10, color='red', alpha=0.2, label='Porpoising Band (5-10 Hz)')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Insufficient High-Speed Straight Telemetry for FFT", ha='center', va='center')
    ax.set_title(f"Porpoising FFT Spectrum ({driver_item['label']})", pad=4)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("FFT Amplitude")
    ax.grid(True, linestyle=':', alpha=0.6)

def draw_platform_pitch_roll(ax, driver_item):
    df = driver_item['data']
    ax.plot(df['Distance'], df['Pitch_Angle_deg'], color='purple', label='Pitch [°]')
    ax.plot(df['Distance'], df['Roll_Angle_deg'], color='darkorange', label='Roll [°]')
    ax.set_title(f"Platform Pitch & Roll Dynamics ({driver_item['label']})", pad=4)
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Platform Angle [°]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=7)

def draw_drs_efficiency(ax, driver_item):
    df = driver_item['data']
    if 'DRS' in df.columns:
        drs_on = df[df['DRS'] >= 8]
        drs_off = df[(df['DRS'] < 8) & (df['Speed'] > 200)]
        ax.scatter(drs_off['Speed'], drs_off['ax_g'], color='gray', s=10, label='DRS Closed', alpha=0.5)
        ax.scatter(drs_on['Speed'], drs_on['ax_g'], color='green', s=12, label='DRS Open')
    else:
        drs_proxy = (df['Speed'] > 250) & (df['Throttle'] > 95)
        ax.scatter(df[~drs_proxy]['Speed'], df[~drs_proxy]['ax_g'], color='gray', s=10, alpha=0.5, label='DRS Closed')
        ax.scatter(df[drs_proxy]['Speed'], df[drs_proxy]['ax_g'], color='green', s=12, label='DRS Open (Est)')
    ax.set_title(f"DRS Efficiency ({driver_item['label']})", pad=4)
    ax.set_xlabel("Speed [km/h]")
    ax.set_ylabel("Longitudinal Accel [g]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=7)

def draw_brake_migration_vs_aero(ax, driver_item):
    df = driver_item['data']
    brake_mask = df['Brake'] > 10.0 if 'Brake' in df.columns else df['ax_g'] < -0.5
    braking_df = df[brake_mask]
    
    if not braking_df.empty:
        sc = ax.scatter(braking_df['ax_g'], braking_df['RideHeight_Front_mm'], c=braking_df['CoP_Front_Pct'], cmap='YlOrRd', s=15)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("CoP Front %", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No Hard Braking Phases Found", ha='center', va='center')
    ax.set_title(f"Brake Entry vs Front RH & Aero ({driver_item['label']})", pad=4)
    ax.set_xlabel("Braking Accel [g]")
    ax.set_ylabel("Front Ride Height [mm]")
    ax.grid(True, linestyle=':', alpha=0.6)

def draw_speed_map(ax, driver_item):
    df = driver_item['data']
    label = driver_item['label']
    if 'X' in df.columns and 'Y' in df.columns:
        x, y, speed = df['X'].values, df['Y'].values, df['Speed'].values
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(vmin=speed.min(), vmax=speed.max())
        lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=3)
        lc.set_array(speed)
        
        ax.add_collection(lc)
        ax.set_facecolor('black')
        ax.axis('equal')
        ax.axis('off')
        ax.set_title(f"Speed Map ({label})", pad=4)
        
        cbar = plt.colorbar(lc, ax=ax, orientation='horizontal', pad=0.03, fraction=0.046)
        cbar.set_label("Speed [km/h]", color='black', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    else:
        ax.text(0.5, 0.5, "GPS Data Unavailable", ha='center', va='center')

def draw_speed_delta_map(ax, runs_list):
    if len(runs_list) < 2:
        ax.text(0.5, 0.5, "Requires >= 2 drivers for Delta Map", ha='center', va='center')
        return
    df1, df2 = runs_list[0]['data'], runs_list[1]['data']
    label1, label2 = runs_list[0]['label'], runs_list[1]['label']
    
    if 'X' in df1.columns and 'Y' in df1.columns:
        x, y = df1['X'].values, df1['Y'].values
        speed2_interp = np.interp(df1['Distance'].values, df2['Distance'].values, df2['Speed'].values)
        delta = df1['Speed'].values - speed2_interp
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(vmin=-15, vmax=15)
        lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidth=3)
        lc.set_array(delta)
        
        ax.add_collection(lc)
        ax.set_facecolor('black')
        ax.axis('equal')
        ax.axis('off')
        ax.set_title(f"Speed Delta: {label1} vs {label2}", pad=4)
        
        cbar = plt.colorbar(lc, ax=ax, orientation='horizontal', pad=0.03, fraction=0.046)
        cbar.set_label(f"Speed Delta [km/h] ({label1} relative to {label2})", color='black', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    else:
        ax.text(0.5, 0.5, "GPS Data Unavailable", ha='center', va='center')

def draw_fsae_chart(ax, runs_list):
    labels = [r['label'] for r in runs_list]
    peak_df = [r['data']['Downforce_N'].max() for r in runs_list]
    ax.bar(labels, peak_df, color='crimson')
    ax.set_ylabel("Downforce [N]")
    ax.set_title("Peak Downforce Comparison", pad=4)
    ax.grid(True, linestyle=':', alpha=0.5)

def draw_cfd_chart(ax, df1, df2):
    df1['X_bins'] = pd.cut(df1['X'], bins=20)
    df2['X_bins'] = pd.cut(df2['X'], bins=20)
    b1 = df1.groupby('X_bins', observed=False)['dFz_N'].sum().reset_index()
    b2 = df2.groupby('X_bins', observed=False)['dFz_N'].sum().reset_index()
    centers = [b.mid for b in b1['X_bins']]
    
    ax.plot(centers, b1['dFz_N'], color='crimson', marker='o', label='Config A')
    ax.plot(centers, b2['dFz_N'], color='royalblue', marker='s', linestyle='--', label='Config B')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel("X-Position [m]")
    ax.set_ylabel("dFz [N]")
    ax.set_title("CFD / Tunnel Downforce Profile", pad=4)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=7)



class InteractiveDashboard:
    """Pop-up figure dashboard featuring an overlay selection modal that floats on top of all subplots."""
    
    def __init__(self, runs_list, session_info, available_views, mode='F1', df_mesh_b=None):
        self.runs_list = runs_list
        self.session_info = session_info
        self.available_views = available_views
        self.mode = mode
        self.df_mesh_b = df_mesh_b
        
        self.driver_names = [r['label'] for r in runs_list]
        self.selected_driver_idx = 0
        
        self.active_features = {view: (i < 4) for i, view in enumerate(available_views)}

  
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.patch.set_facecolor('#f8f9fa')

    
        self.ax_btn_select = self.fig.add_axes([0.05, 0.015, 0.22, 0.04])
        self.ax_btn_select.set_zorder(101)
        
        self.btn_select = Button(self.ax_btn_select, "Select Analysis Views", color='#e9ecef', hovercolor='#ced4da')
        self.btn_select.on_clicked(self.toggle_modal_menu)

        self.ax_radio = self.fig.add_axes([0.78, 0.012, 0.18, 0.045], facecolor='#e9ecef')
        self.ax_radio.set_zorder(101)
        self.ax_radio.set_title("Active Driver / Run", fontsize=7, fontweight='bold')
        self.radio_widget = RadioButtons(self.ax_radio, labels=self.driver_names, active=0, activecolor='dodgerblue')
        self.radio_widget.on_clicked(self.on_driver_change)

      
        self.ax_modal = self.fig.add_axes([0.05, 0.07, 0.28, 0.65], facecolor='#ffffff')
        self.ax_modal.set_zorder(100)
        self.ax_modal.set_title("Toggle Analysis Views", fontsize=8, fontweight='bold', pad=4)
        
        self.check_widget = CheckButtons(
            ax=self.ax_modal,
            labels=list(self.active_features.keys()),
            actives=list(self.active_features.values())
        )
        self.check_widget.on_clicked(self.on_check_toggle)
        self.ax_modal.set_visible(False)

        self.plot_axes = []
        self.render_dashboard()
        plt.show()

    def toggle_modal_menu(self, event):
        is_visible = self.ax_modal.get_visible()
        self.ax_modal.set_visible(not is_visible)
        self.fig.canvas.draw_idle()

    def on_check_toggle(self, label):
        self.active_features[label] = not self.active_features[label]
        self.render_dashboard()

    def on_driver_change(self, label):
        self.selected_driver_idx = self.driver_names.index(label)
        self.render_dashboard()

    def clear_plot_axes(self):
        keep_axes = {self.ax_btn_select, self.ax_radio, self.ax_modal}
        for ax in list(self.fig.axes):
            if ax not in keep_axes:
                self.fig.delaxes(ax)
        self.plot_axes.clear()

    def render_dashboard(self):
        self.clear_plot_axes()
        active_plots = [k for k, v in self.active_features.items() if v]

        if not active_plots:
            ax = self.fig.add_axes([0.1, 0.12, 0.8, 0.78])
            ax.set_zorder(1)
            ax.text(0.5, 0.5, "No views selected.\nClick 'Select Analysis Views' at the bottom to choose subplots.",
                    ha='center', va='center', fontsize=12, color='gray')
            self.plot_axes.append(ax)
            self.fig.canvas.draw_idle()
            return

        n_plots = len(active_plots)
        cols = 3 if n_plots > 4 else (2 if n_plots > 1 else 1)
        rows = (n_plots + cols - 1) // cols

        top_margin = 0.88
        bottom_margin = 0.10
        left_margin = 0.06
        right_margin = 0.96
        
        avail_w = right_margin - left_margin
        avail_h = top_margin - bottom_margin
        
        h_gap = 0.05
        v_gap = 0.09 if rows > 1 else 0.0
        
        col_width = (avail_w - (cols - 1) * h_gap) / cols
        row_height = (avail_h - (rows - 1) * v_gap) / rows

        current_driver = self.runs_list[self.selected_driver_idx]

        for idx, view in enumerate(active_plots):
            r = idx // cols
            c = idx % cols
            
            left = left_margin + c * (col_width + h_gap)
            bottom = top_margin - (r + 1) * row_height - r * v_gap

            ax = self.fig.add_axes([left, bottom, col_width, row_height])
            ax.set_zorder(1)
            self.plot_axes.append(ax)

            if view == 'Telemetry & Speed':
                draw_telemetry_overlay(ax, self.runs_list, self.selected_driver_idx)
            elif view == 'CoP & Understeer Gradient':
                draw_cop_understeer(ax, current_driver)
            elif view == 'Yaw & Aero Sensitivity':
                draw_yaw_sensitivity(ax, current_driver)
            elif view == 'Coast-Down Drag (CdA)':
                draw_coastdown_cda(ax, current_driver)
            elif view == 'Top Speed Clipping':
                draw_top_speed_clipping(ax, current_driver)
            elif view == 'Aero-Banded G-G Diagram':
                draw_aero_banded_gg(ax, current_driver)
            elif view == 'Corner Speed Profiling':
                draw_corner_profiling(ax, current_driver)
            elif view == 'Porpoising FFT Analysis':
                draw_porpoising_fft(ax, current_driver)
            elif view == 'Platform Pitch & Roll':
                draw_platform_pitch_roll(ax, current_driver)
            elif view == 'DRS Efficiency':
                draw_drs_efficiency(ax, current_driver)
            elif view == 'Brake Migration vs Aero':
                draw_brake_migration_vs_aero(ax, current_driver)
            elif view == 'Speed Map':
                draw_speed_map(ax, current_driver)
            elif view == 'Speed Delta Map':
                draw_speed_delta_map(ax, self.runs_list)
            elif view == 'Setup Chart':
                draw_fsae_chart(ax, self.runs_list)
            elif view == 'CFD Analysis':
                draw_cfd_chart(ax, self.runs_list[0]['data'], self.df_mesh_b if self.df_mesh_b is not None else self.runs_list[0]['data'])

        self.ax_modal.set_zorder(100)
        self.ax_btn_select.set_zorder(101)
        self.ax_radio.set_zorder(101)

        self.fig.suptitle(f"Motorsport Vehicle Dynamics & Aerodynamic Dashboard\n{self.session_info}", fontsize=11, fontweight='bold', y=0.96)
        self.fig.canvas.draw_idle()



def run_f1_analysis():
    print("\n--- FASTF1 SESSION CONFIGURATOR (Enter 'M' at any prompt to return to Main Menu) ---")
    
    year_str = get_input("Enter Season Year (2018-2025) [Default 2023]: ", "2023")
    if year_str == 'MAIN_MENU': return
    year = int(year_str)

    gp = get_input("Enter Grand Prix Name [Default 'Silverstone']: ", "Silverstone")
    if gp == 'MAIN_MENU': return

    sess_type = get_input("Enter Session Type (Q, R, FP1) [Default 'Q']: ", "Q").upper()
    if sess_type == 'MAIN_MENU': return

    num_str = get_input("How many drivers would you like to compare? (1, 2, or 3) [Default 2]: ", "2")
    if num_str == 'MAIN_MENU': return
    num_drivers = int(num_str)

    defaults = ['VER', 'LEC', 'NOR']
    driver_codes = []
    for i in range(num_drivers):
        code = get_input(f"Enter Driver {i+1} Code [Default '{defaults[i]}']: ", defaults[i]).upper()
        if code == 'MAIN_MENU': return
        driver_codes.append(code)

    print(f"\n[INFO] Loading {year} {gp} [{sess_type}] data...")
    session = fastf1.get_session(year, gp, sess_type)
    session.load(telemetry=True, laps=True)
    
    runs_list = []
    for code in driver_codes:
        laps = session.laps.pick_drivers(code)
        if not laps.empty:
            tel = laps.pick_fastest().get_telemetry().add_distance()
            df_proc = AeroDynamicsEngine(tel).process_all()
            runs_list.append({'label': code, 'data': df_proc})

    if not runs_list:
        print("[ERROR] No valid timing data loaded.")
        return

    available_views = [
        'Telemetry & Speed',
        'CoP & Understeer Gradient',
        'Yaw & Aero Sensitivity',
        'Coast-Down Drag (CdA)',
        'Top Speed Clipping',
        'Aero-Banded G-G Diagram',
        'Corner Speed Profiling',
        'Porpoising FFT Analysis',
        'Platform Pitch & Roll',
        'DRS Efficiency',
        'Brake Migration vs Aero',
        'Speed Map',
        'Speed Delta Map'
    ]
    sess_str = f"{year} {gp} {sess_type}"
    InteractiveDashboard(runs_list, sess_str, available_views, mode='F1')


def run_fsae_motec_analysis():
    print("\n--- FSAE / MOTEC i2 PRO CONFIGURATOR (Enter 'M' at any prompt to return to Main Menu) ---")
    
    mass_str = get_input("Enter Mass + Driver (kg) [Default 230]: ", "230.0")
    if mass_str == 'MAIN_MENU': return
    mass = float(mass_str)

    wb_str = get_input("Enter Wheelbase (m) [Default 1.535]: ", "1.535")
    if wb_str == 'MAIN_MENU': return
    wb = float(wb_str)

    tw_str = get_input("Enter Track Width (m) [Default 1.200]: ", "1.200")
    if tw_str == 'MAIN_MENU': return
    tw = float(tw_str)

    kf_str = get_input("Enter Front Stiffness (N/m) [Default 45000]: ", "45000.0")
    if kf_str == 'MAIN_MENU': return
    k_f = float(kf_str)

    kr_str = get_input("Enter Rear Stiffness (N/m) [Default 50000]: ", "50000.0")
    if kr_str == 'MAIN_MENU': return
    k_r = float(kr_str)

    num_str = get_input("How many FSAE setups/runs to compare? (1, 2, or 3) [Default 2]: ", "2")
    if num_str == 'MAIN_MENU': return
    num_setups = int(num_str)

    runs_list = []
    for i in range(num_setups):
        setup_name = get_input(f"Enter Name for Setup {i+1} [Default 'Setup_{i+1}']: ", f"Setup_{i+1}")
        if setup_name == 'MAIN_MENU': return
        
        cla_val = 2.8 + (i * 0.4)
        dist = np.linspace(0, 800, 400)
        time_pts = pd.date_range(start='2026-01-01', periods=400, freq='20ms')
        speed = 30 + 45 * np.sin(dist / (40 + i*5))**2
        
        df_raw = pd.DataFrame({
            'Time': time_pts, 'Distance': dist, 'Speed': speed,
            'Steer_Angle': 12 * np.sin(dist / 30),
            'Throttle': np.where(speed > 50, 100, 20),
            'Brake': np.where(speed < 40, 50, 0)
        })

        engine = AeroDynamicsEngine(df_raw, mass_kg=mass, wheelbase_m=wb, track_width_m=tw,
                                    k_front=k_f, k_rear=k_r, cla=cla_val, static_rh_front=28.0, static_rh_rear=32.0)
        runs_list.append({'label': setup_name, 'data': engine.process_all()})

    available_views = ['Telemetry & Speed', 'Setup Chart', 'Aero-Banded G-G Diagram', 'Platform Pitch & Roll', 'CoP & Understeer Gradient']
    InteractiveDashboard(runs_list, "FSAE Session", available_views, mode='FSAE')


def run_star_ccm_analysis():
    print("\n--- SIEMENS STAR-CCM+ CFD ANALYSIS MODULE ---")
    proc = STARCCM_CFD_Processor()
    df_mesh_a = proc.load_and_process(seed=42, df_bias=1.0)
    df_mesh_b = proc.load_and_process(seed=101, df_bias=1.25)
    
    print("[INFO] Launching STAR-CCM+ Interactive CFD Visualizer...")
    runs_list = [{'label': 'Config A (Baseline)', 'data': df_mesh_a}]
    InteractiveDashboard(runs_list, "CFD Mesh Analysis", ['CFD Analysis'], mode='CFD', df_mesh_b=df_mesh_b)


def run_wind_tunnel_analysis():
    print("\n--- FSAE WIND TUNNEL DATA IMPORT MODULE (Enter 'M' to return) ---")
    print("1. Surface Pressure Tap Grid (3D Points: X, Y, Z, Pressure/Cp, Area)")
    print("2. Time-Series / Speed Sweep Telemetry CSV (Speed, Throttle, Steer, etc.)")
    print("3. Aero Balance & Load Cell Summary (Tunnel CLA, CDA & Ride Height Parameters)")
    
    choice = get_input("\nSelect Import Type (1-3) [Default 1]: ", "1")
    if choice == 'MAIN_MENU': return

    if choice == '1':
        path = get_input("Enter CSV file path for Pressure Tap Data [Default 'wind_tunnel_surface.csv']: ", "wind_tunnel_surface.csv")
        if path == 'MAIN_MENU': return
        
        proc = STARCCM_CFD_Processor(filepath=path if os.path.exists(path) else None)
        if not os.path.exists(path):
            print(f"[NOTE] File '{path}' not found. Generating sample wind tunnel surface mesh...")
        
        df_mesh = proc.load_and_process(seed=77, df_bias=1.1)
        runs_list = [{'label': 'Wind Tunnel Surface Mesh', 'data': df_mesh}]
        
        df_baseline = STARCCM_CFD_Processor().load_and_process(seed=42, df_bias=1.0)
        InteractiveDashboard(runs_list, "Wind Tunnel Pressure Tap Analysis", ['CFD Analysis'], mode='CFD', df_mesh_b=df_baseline)

    elif choice == '2':
        path = get_input("Enter CSV file path for Telemetry/Sweep Data [Default 'wind_tunnel_sweep.csv']: ", "wind_tunnel_sweep.csv")
        if path == 'MAIN_MENU': return

        if os.path.exists(path):
            df_raw = pd.read_csv(path)
            print(f"[INFO] Successfully imported '{path}'.")
            if 'Distance' not in df_raw.columns:
                df_raw['Distance'] = np.linspace(0, 1000, len(df_raw))
        else:
            print(f"[NOTE] File '{path}' not found. Generating synthetic wind tunnel velocity sweep...")
            dist = np.linspace(0, 1000, 500)
            speed = np.linspace(40, 140, 500)
            df_raw = pd.DataFrame({
                'Time': pd.date_range(start='2026-01-01', periods=500, freq='20ms'),
                'Distance': dist,
                'Speed': speed,
                'Steer_Angle': np.zeros(500),
                'Throttle': np.full(500, 100),
                'Brake': np.zeros(500)
            })

        print("\n--- VEHICLE DYNAMICS PARAMETERS ---")
        mass_str = get_input("Enter Mass + Driver (kg) [Default 230]: ", "230.0")
        if mass_str == 'MAIN_MENU': return
        wb_str = get_input("Enter Wheelbase (m) [Default 1.535]: ", "1.535")
        if wb_str == 'MAIN_MENU': return
        tw_str = get_input("Enter Track Width (m) [Default 1.200]: ", "1.200")
        if tw_str == 'MAIN_MENU': return
        kf_str = get_input("Enter Front Stiffness (N/m) [Default 45000]: ", "45000.0")
        if kf_str == 'MAIN_MENU': return
        kr_str = get_input("Enter Rear Stiffness (N/m) [Default 50000]: ", "50000.0")
        if kr_str == 'MAIN_MENU': return

        print("\n--- AERODYNAMIC PARAMETERS ---")
        cla_str = get_input("Enter Wind Tunnel CLA (CdA/CLA) [Default 3.2]: ", "3.2")
        if cla_str == 'MAIN_MENU': return
        cda_str = get_input("Enter Wind Tunnel CDA [Default 1.1]: ", "1.1")
        if cda_str == 'MAIN_MENU': return

        engine = AeroDynamicsEngine(
            df_raw, 
            mass_kg=float(mass_str),
            wheelbase_m=float(wb_str),
            track_width_m=float(tw_str),
            k_front=float(kf_str),
            k_rear=float(kr_str),
            cla=float(cla_str), 
            cda=float(cda_str)
        )
        df_processed = engine.process_all()
        runs_list = [{'label': 'Wind Tunnel Run', 'data': df_processed}]

        available_views = ['Telemetry & Speed', 'CoP & Understeer Gradient', 'Coast-Down Drag (CdA)', 'Platform Pitch & Roll']
        InteractiveDashboard(runs_list, "Wind Tunnel Telemetry Sweep", available_views, mode='FSAE')

    elif choice == '3':
        print("\n--- VEHICLE DYNAMICS PARAMETERS ---")
        mass_str = get_input("Enter Mass + Driver (kg) [Default 230]: ", "230.0")
        if mass_str == 'MAIN_MENU': return
        wb_str = get_input("Enter Wheelbase (m) [Default 1.535]: ", "1.535")
        if wb_str == 'MAIN_MENU': return
        tw_str = get_input("Enter Track Width (m) [Default 1.200]: ", "1.200")
        if tw_str == 'MAIN_MENU': return
        kf_str = get_input("Enter Front Stiffness (N/m) [Default 45000]: ", "45000.0")
        if kf_str == 'MAIN_MENU': return
        kr_str = get_input("Enter Rear Stiffness (N/m) [Default 50000]: ", "50000.0")
        if kr_str == 'MAIN_MENU': return

        print("\n--- AERODYNAMIC PARAMETERS ---")
        cla_str = get_input("Enter Base Tunnel CLA [Default 3.5]: ", "3.5")
        if cla_str == 'MAIN_MENU': return
        cda_str = get_input("Enter Base Tunnel CDA [Default 1.15]: ", "1.15")
        if cda_str == 'MAIN_MENU': return
        rh_f_str = get_input("Enter Front Static Ride Height (mm) [Default 25.0]: ", "25.0")
        if rh_f_str == 'MAIN_MENU': return
        rh_r_str = get_input("Enter Rear Static Ride Height (mm) [Default 45.0]: ", "45.0")
        if rh_r_str == 'MAIN_MENU': return

        dist = np.linspace(0, 800, 400)
        speed = 30 + 50 * np.sin(dist / 50)**2
        df_raw = pd.DataFrame({
            'Time': pd.date_range(start='2026-01-01', periods=400, freq='20ms'),
            'Distance': dist,
            'Speed': speed,
            'Steer_Angle': 8 * np.sin(dist / 25),
            'Throttle': np.where(speed > 45, 90, 10),
            'Brake': np.where(speed < 35, 40, 0)
        })

        engine = AeroDynamicsEngine(
            df_raw, 
            mass_kg=float(mass_str),
            wheelbase_m=float(wb_str),
            track_width_m=float(tw_str),
            k_front=float(kf_str),
            k_rear=float(kr_str),
            cla=float(cla_str), 
            cda=float(cda_str), 
            static_rh_front=float(rh_f_str), 
            static_rh_rear=float(rh_r_str)
        )
        df_processed = engine.process_all()
        runs_list = [{'label': 'Tunnel Load Cell Setup', 'data': df_processed}]

        available_views = ['Setup Chart', 'Aero-Banded G-G Diagram', 'CoP & Understeer Gradient', 'Platform Pitch & Roll']
        InteractiveDashboard(runs_list, "Wind Tunnel Aero Balance Setup", available_views, mode='FSAE')


def main_menu():
    while True:
        print("\n" + "="*55)
        print("   MOTORSPORT AERODYNAMIC & HANDLING ANALYSIS TOOL   ")
        print("="*55)
        print("1. Compare 1-3 F1 Drivers (Telemetry, Ride Height & Platform Dynamics)")
        print("2. Import / Compare FSAE MoTeC Setups (Custom Vehicle Criteria)")
        print("3. Run STAR-CCM+ 3D Mesh Pressure & Load Comparison Chart")
        print("4. Import & Analyze FSAE Wind Tunnel CSV Data")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        if choice == '1':
            run_f1_analysis()
        elif choice == '2':
            run_fsae_motec_analysis()
        elif choice == '3':
            run_star_ccm_analysis()
        elif choice == '4':
            run_wind_tunnel_analysis()
        elif choice == '5':
            sys.exit()

if __name__ == '__main__':
    main_menu()
