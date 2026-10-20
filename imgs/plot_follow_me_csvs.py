#!/usr/bin/env python3
"""Plot leader and follower trajectories and velocities from recorded CSV logs."""

from argparse import ArgumentParser
from csv import DictReader
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_CONTROLLERS = ('mlp', 'svm', 'lstm', 'pd')


def load_rows(csv_path):
    leader = {'time': [], 'x': [], 'y': [], 'linear': [], 'angular': []}
    follower = {'time': [], 'x': [], 'y': [], 'linear': [], 'angular': []}

    with csv_path.open(newline='', encoding='utf-8') as file:
        reader = DictReader(file)
        for row in reader:
            target = leader if row['source'] == 'leader' else follower
            target['time'].append(float(row['time_s']))
            target['x'].append(float(row['x_m']))
            target['y'].append(float(row['y_m']))
            target['linear'].append(float(row['linear_mps']))
            target['angular'].append(float(row['angular_rps']))

    return leader, follower


def derivative_series(times, values):
    derivative_times = []
    derivatives = []
    previous_time = None
    previous_value = None
    for time, value in zip(times, values):
        if previous_time is None:
            previous_time = time
            previous_value = value
            continue
        dt = time - previous_time
        if dt > 0.0:
            derivative_times.append(time)
            derivatives.append((value - previous_value) / dt)
        previous_time = time
        previous_value = value
    return derivative_times, derivatives


def mean_absolute_value(values):
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def average_abs_acceleration_from_speed(times, speeds):
    _, accelerations = derivative_series(times, speeds)
    return mean_absolute_value(accelerations)


def acceleration_series_from_speed(times, speeds):
    _, accelerations = derivative_series(times, speeds)
    return accelerations


def jerk_stats_from_speed(times, speeds):
    acceleration_times, accelerations = derivative_series(times, speeds)
    if not accelerations:
        return 0.0, 0.0
    _, jerks = derivative_series(acceleration_times, accelerations)
    if not jerks:
        return 0.0, 0.0
    mean_jerk = sum(jerks) / len(jerks)
    mean_abs_jerk = mean_absolute_value(jerks)
    return mean_jerk, mean_abs_jerk


def build_parser():
    parser = ArgumentParser(
        description='Plot leader reference and all controller follower CSV logs.')
    parser.add_argument(
        '--csv-dir',
        default='/home/sahar/workspaces/follow_me_ros2/src/follow_me_ros2/imgs',
        help='Directory containing follow_me_<controller>.csv files',
    )
    parser.add_argument(
        '--controllers',
        nargs='+',
        default=list(DEFAULT_CONTROLLERS),
        help='Controller names to compare',
    )
    parser.add_argument(
        '--reference-controller',
        default='pd',
        help='Controller CSV to use as the leader reference',
    )
    parser.add_argument(
        '--output',
        default='',
        help='Path to save the comparison figure; defaults to csv-dir/follow_me_comparison.pdf',
    )
    return parser


def plot_comparison(csv_dir, controllers, reference_controller, output_path):
    csv_dir = Path(csv_dir)
    reference_csv = csv_dir / f'follow_me_{reference_controller}.csv'
    if not reference_csv.exists():
        raise FileNotFoundError(f'Missing reference CSV: {reference_csv}')

    records = {}
    for controller in controllers:
        csv_path = csv_dir / f'follow_me_{controller}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f'Missing controller CSV: {csv_path}')
        records[controller] = load_rows(csv_path)

    leader, _ = records[reference_controller]

    def display_name(controller):
        return 'baseline' if controller == reference_controller else controller.upper()

    figure, axes = plt.subplots(2, 1, figsize=(9, 12))
    trajectory_axis, acceleration_axis = axes
    colors = plt.cm.tab10.colors

    def controller_color(controller, index):
        if controller == reference_controller:
            return '#FFD700'
        return {
            'mlp': 'tab:blue',
            'lstm': 'tab:green',
            'svm': 'tab:red',
        }.get(controller, colors[index % len(colors)])

    trajectory_axis.plot(
        leader['x'],
        leader['y'],
        color='black',
        linewidth=3.0,
        label='Leader',
    )
    for index, controller in enumerate(controllers):
        _, follower = records[controller]
        trajectory_axis.plot(
            follower['x'],
            follower['y'],
            color=controller_color(controller, index),
            linewidth=1.8,
            label=display_name(controller),
        )
        trajectory_axis.scatter(
            follower['x'][0],
            follower['y'][0],
            color='green',
            marker='^',
            s=80,
            edgecolor='black',
            zorder=5,
            label='Follower start' if index == 0 else '_nolegend_',
        )
    trajectory_axis.scatter(
        leader['x'][0],
        leader['y'][0],
        color='green',
        marker='o',
        s=90,
        edgecolor='black',
        zorder=5,
        label='Leader start',
    )
    trajectory_axis.scatter(
        leader['x'][-1],
        leader['y'][-1],
        color='red',
        marker='X',
        s=110,
        edgecolor='black',
        zorder=5,
        label='Stop',
    )
    # trajectory_axis.set_title('Trajectory')
    trajectory_axis.set_xlabel('x [m]')
    trajectory_axis.set_ylabel('y [m]')
    trajectory_axis.set_aspect('equal', adjustable='box')
    trajectory_axis.set_xlim(-6.0, 0.5)
    trajectory_axis.grid(True)
    trajectory_axis.legend(prop={'size': 13, 'weight': 'bold'})

    labels = [display_name(controller) for controller in controllers]
    linear_avg_accelerations = []
    angular_avg_accelerations = []
    for controller in controllers:
        _, follower = records[controller]
        linear_avg_accelerations.append(
            average_abs_acceleration_from_speed(
                follower['time'],
                follower['linear'],
            )
        )
        angular_avg_accelerations.append(
            average_abs_acceleration_from_speed(
                follower['time'],
                follower['angular'],
            )
        )

    positions = list(range(len(labels)))
    bar_width = 0.25
    bar_colors = [
        controller_color(controller, index)
        for index, controller in enumerate(controllers)
    ]
    linear_bars = acceleration_axis.bar(
        [position - bar_width / 2.0 for position in positions],
        linear_avg_accelerations,
        width=bar_width,
        color=bar_colors,
        edgecolor='black',
        label='Linear [m/s²]',
    )
    angular_bars = acceleration_axis.bar(
        [position + bar_width / 2.0 for position in positions],
        angular_avg_accelerations,
        width=bar_width,
        color=bar_colors,
        edgecolor='black',
        hatch='//',
        label='Angular [rad/s²]',
    )

    # acceleration_axis.set_title('Average linear and angular acceleration')
    acceleration_axis.set_ylabel('Average acceleration')
    acceleration_axis.set_xticks(positions)
    acceleration_axis.set_xticklabels(labels, rotation=20)
    acceleration_axis.grid(True, axis='y')
    acceleration_axis.legend(prop={'size': 13, 'weight': 'bold'})

    for bars, values in (
        (linear_bars, linear_avg_accelerations),
        (angular_bars, angular_avg_accelerations),
    ):
        for bar, value in zip(bars, values):
            acceleration_axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
            )

    for axis in axes:
        axis.set_title(axis.get_title(), fontsize=18, fontweight='bold')
        axis.xaxis.label.set_size(15)
        axis.xaxis.label.set_weight('bold')
        axis.yaxis.label.set_size(15)
        axis.yaxis.label.set_weight('bold')
        axis.tick_params(axis='both', labelsize=15, width=1.5)
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            tick_label.set_fontweight('bold')
        for spine in axis.spines.values():
            spine.set_linewidth(1.5)

    figure.tight_layout()

    trajectory_position = trajectory_axis.get_position()
    acceleration_position = acceleration_axis.get_position()
    acceleration_axis.set_position([
        trajectory_position.x0,
        acceleration_position.y0,
        trajectory_position.width,
        acceleration_position.height,
    ])

    if not output_path:
        output_path = csv_dir / 'follow_me_comparison.pdf'
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    print(f'Saved comparison figure to {output_path}')


def main():
    args = build_parser().parse_args()
    plot_comparison(
        args.csv_dir,
        args.controllers,
        args.reference_controller,
        args.output,
    )


if __name__ == '__main__':
    main()
