# SOMA (Self-Organizing Memory Architecture) Experiments

![SOMA Logo](SOMA-logo-xs.png)




This repository contains a collection of experiments for the SOMA (Self-Organizing Memory Architecture) project. SOMA is a cognitive architecture that implements self-organizing memory units for adaptive agent behavior.

## Project Structure

The project is organized into several experiment directories, each focusing on different aspects of the SOMA architecture:

### Coherence Benchmark

Located in `experiments/coherence_benchmark/`, this experiment evaluates the coherence of conversations using the SOMA architecture. It implements:

- A core SOMA implementation with LangGraph integration
- Coherence-Onboarding Protocol (COP) for the first 10 interactions
- Coherence evaluation metrics and benchmarking

### Color Drift RL Task Benchmark

Located in `experiments/color_drift_rl_task_benchmark/`, this experiment implements a reinforcement learning environment with color drift to test adaptive behavior. It features:

- A torus-based environment with walls and gems
- Color drift mechanics that require adaptive strategies
- Comparison between SOMA-based agents and traditional RL approaches
- Statistical analysis and visualization of performance

### Hidden Tasks Metacognitive Evaluation

Located in `experiments/hidden_tasks_metacognitive/`, this experiment evaluates metacognitive capabilities in SOMA. It includes:

- Information-theoretic analysis of metacognitive learning dynamics
- Comprehensive statistical analysis of performance
- Strategic innovation and pattern analysis
- Visualization of temporal dynamics

## Getting Started

### Prerequisites

This project requires Python 3.8+ and several dependencies. Install all required packages using:

```bash
pip install -r requirements.txt
```

### Running Experiments

Each experiment can be run independently from its respective directory:

#### Coherence Benchmark

```bash
cd experiments/coherence_benchmark
python soma_core.py
```

#### Color Drift RL Task Benchmark

Open and run the Jupyter notebook:

```bash
cd experiments/color_drift_rl_task_benchmark
jupyter notebook color-drift.ipynb
```

#### Hidden Tasks Metacognitive

Open and run the Jupyter notebook:

```bash
cd experiments/hidden_tasks_metacognitive
jupyter notebook metacog_evaluation.ipynb
```

## Key Features

- **Self-Organizing Memory Units**: Implementation of Executable Memory Units (EMUs) that adapt to different tasks
- **Metacognitive Capabilities**: Evaluation of metacognitive learning and strategic adaptation
- **Coherence Protocols**: Mechanisms for maintaining conversational coherence
- **Reinforcement Learning Integration**: Combination of symbolic reasoning with reinforcement learning
- **Comprehensive Benchmarking**: Statistical analysis and visualization of performance metrics

## Results

The experiments demonstrate:

1. **Coherence Benchmark**: How SOMA maintains conversational coherence through adaptive memory structures
2. **Color Drift RL**: How SOMA-enhanced agents outperform traditional RL in environments requiring adaptation
3. **Metacognitive Evaluation**: How metacognitive capabilities emerge and improve performance in complex tasks

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

Copyright © 2025 Cadenzai, Inc.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### Network Use

If you modify this program and run it on a server that users can interact with over a network, you must provide those users with access to the source code of your modified version.

## Acknowledgments

- The SOMA architecture builds on research in cognitive architectures and adaptive agents
- Visualization techniques adapted from standard scientific visualization libraries
