"""
Command-Line Interface and Rich Dashboard for AIX-Evolver.
"""

import argparse
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from aix_evolver.__version__ import __version__
from aix_evolver.dataset import ProblemManager
from aix_evolver.generator import RolloutGenerator
from aix_evolver.verifier import ReasoningVerifier
from aix_evolver.elastic_brain import ElasticBrain
from aix_evolver.trainer import ElasticTrainer

console = Console()


def render_banner():
    console.print(
        Panel.fit(
            f"[bold cyan]AIX-Evolver v{__version__}[/bold cyan] — [bold green]Elastic Reasoning Self-Improvement Engine[/bold green]\n"
            f"[dim]Created by Alex Pfeil (@pfolex) | Frugal Autonomous Neural Evolution[/dim]",
            border_style="cyan"
        )
    )


def run_status():
    render_banner()
    brain = ElasticBrain()

    table = Table(title="Elastic Brain Parameter Economy", border_style="cyan")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="bold white")

    table.add_row("Base Architecture", "1.5B (DeepSeek-R1 Distill + Qwen SLERP)")
    table.add_row("Base Parameters", f"{brain.base_params:,}")
    table.add_row("Active Modular Experts", str(len([e for e in brain.modular_experts.values() if e.is_active])))
    table.add_row("Added Footprint", f"{brain.added_parameter_footprint_mb:.2f} MB")
    table.add_row("Total Active Parameters", f"{brain.total_active_parameters:,}")

    console.print(table)

    if brain.modular_experts:
        exp_table = Table(title="Active Modular Experts", border_style="green")
        exp_table.add_column("Expert Name", style="bold cyan")
        exp_table.add_column("Target Category", style="magenta")
        exp_table.add_column("LoRA Rank", style="yellow")
        exp_table.add_column("Parameters", style="green")
        exp_table.add_column("Active", style="bold")

        for name, exp in brain.modular_experts.items():
            status = "[green]YES[/green]" if exp.is_active else "[red]PRUNED[/red]"
            exp_table.add_row(name, exp.target_category, str(exp.lora_rank), f"{exp.params_count:,}", status)
        console.print(exp_table)


def run_evolution(
    steps: int = 5,
    model: str = "aix",
    group_size: int = 3,
    source: str = "curated",
    dry_run: bool = False
):
    render_banner()
    console.print(
        f"[bold cyan]Starting Self-Evolution Loop on model '[bold yellow]{model}[/bold yellow]' "
        f"using source '[bold green]{source}[/bold green]' for {steps} steps...[/bold cyan]\n"
    )

    manager = ProblemManager()
    manager.load_by_source(source=source, limit=steps)
    generator = RolloutGenerator(model_name=model)
    verifier = ReasoningVerifier()
    brain = ElasticBrain()
    trainer = ElasticTrainer(dry_run=dry_run)

    problems = manager.get_batch(count=steps)

    for step_idx, problem in enumerate(problems, 1):
        console.print(f"\n[bold white]═══ Step {step_idx}/{steps}: Problem [cyan]{problem.id}[/cyan] ({problem.category}) ═══[/bold white]")
        console.print(f"[dim]{problem.question}[/dim]")

        # 1. Generate multi-path rollouts
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task(f"Generating {group_size} reasoning paths via Ollama...", total=None)
            rollouts = generator.generate_group(problem.id, problem.question, group_size=group_size)

        # 2. Verify all rollouts
        best_score = None
        best_rollout = None
        worst_rollout = None

        for r in rollouts:
            score = verifier.score_rollout(r.full_text, problem.expected, r.tokens)
            if best_score is None or score.reward > best_score.reward:
                best_score = score
                best_rollout = r
            if worst_rollout is None or score.reward < (worst_rollout[1].reward if worst_rollout else 0):
                worst_rollout = (r, score)

        is_success = best_score.is_correct if best_score else False
        manager.record_attempt(problem.id, is_success)
        brain.record_performance(problem.category, best_score.reward if best_score else -1.0)

        # 3. Display outcome
        status_badge = "[bold green]SOLVED[/bold green]" if is_success else "[bold red]FAILED[/bold red]"
        pred_val = best_score.predicted_answer if best_score else "N/A"
        refl_pct = (best_score.reflection_intensity * 100) if best_score else 0.0

        console.print(
            f"Outcome: {status_badge} | Pred: [bold]{pred_val}[/bold] | True: [bold]{problem.expected}[/bold] "
            f"| Reflection: {refl_pct:.0f}% | Reward: [bold]{best_score.reward:.2f}[/bold]"
        )

        # 4. Check for Capacity Plateau
        is_plateau, plateau_reason = brain.detect_plateau(problem.category)
        if is_plateau:
            console.print(f"[bold yellow]⚠️  Capacity Plateau Detected: {plateau_reason}[/bold yellow]")
            console.print("[bold cyan]🌱 Triggering Just-In-Time LoRA Modular Expansion...[/bold cyan]")
            event = brain.spawn_modular_expert(problem.category, plateau_reason, lora_rank=8)
            console.print(
                f"[bold green]✓ Spawned modular expert '{event.expert_name}' (+{event.added_params:,} params, "
                f"+{event.growth_percent:.2f}% growth, footprint: {brain.added_parameter_footprint_mb:.2f} MB)[/bold green]"
            )

        # 5. Execute 1-step parameter update if successful rollout was found
        if is_success and best_rollout:
            loser_text = worst_rollout[0].full_text if worst_rollout and not worst_rollout[1].is_correct else None
            active_expert = list(brain.modular_experts.keys())[-1] if brain.modular_experts else "frugal_core"
            metrics = trainer.train_on_rollouts(
                problem.id,
                problem.question,
                best_rollout.full_text,
                losing_rollout=loser_text,
                adapter_name=active_expert
            )
            console.print(f"[bold green]🧠 Weights Updated:[/bold green] Step={metrics.step} | Loss={metrics.loss:.4f} | Expert={metrics.adapter_name}")

    console.print("\n[bold cyan]════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold green]✓ Evolution cycle finished. Total parameters: {brain.total_active_parameters:,} (+{brain.added_parameter_footprint_mb:.2f} MB added)[/bold green]\n")


def main():
    parser = argparse.ArgumentParser(description="AIX-Evolver: Elastic Reasoning Self-Improvement Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # evolve command
    evolve_p = subparsers.add_parser("evolve", help="Run autonomous reasoning evolution loop")
    evolve_p.add_argument("--steps", type=int, default=5, help="Number of reasoning steps to evolve")
    evolve_p.add_argument("--model", type=str, default="aix", help="Target Ollama model name")
    evolve_p.add_argument("--group-size", type=int, default=3, help="Number of rollouts per problem")
    evolve_p.add_argument(
        "--source",
        type=str,
        default="curated",
        choices=["curated", "gsm8k", "reasoning-gym", "open-r1"],
        help="Problem source dataset (curated, gsm8k, reasoning-gym, open-r1)"
    )
    evolve_p.add_argument("--dry-run", action="store_true", help="Simulate weight updates for fast validation")

    # status command
    subparsers.add_parser("status", help="Inspect Elastic Brain parameter economy")

    args = parser.parse_args()

    if args.command == "evolve":
        run_evolution(
            steps=args.steps,
            model=args.model,
            group_size=args.group_size,
            source=args.source,
            dry_run=args.dry_run
        )
    elif args.command == "status":
        run_status()
    else:
        render_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
