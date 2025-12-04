from rich.tree import Tree
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live
import sys

from rag import get_response

console = Console()

def print_banner():
    """Display a nice header"""
    banner_text = """
    [bold cyan]🔮 RAG AI Chat Terminal[/bold cyan]
    [dim]Ask questions about your documents. Type 'exit' to quit.[/dim]
    """
    console.print(Panel(banner_text, border_style="blue", expand=False))

def main():
    console.clear()
    print_banner()

    while True:
        try:
            console.print("\n[bold green]👤 Your question:[/bold green]")
            user_input = Prompt.ask("💬")

            if user_input.lower() in ["exit", "quit", "q", "выход"]:
                console.print("\n[bold yellow]👋 Goodbye! Session terminated.[/bold yellow]")
                break

            if not user_input.strip():
                continue

            console.print("")

            with console.status("[bold magenta]🤖 Reading documents and generating answer...[/bold magenta]", spinner="dots"):
                response = get_response(user_input)

            response_text = str(response)
            console.print("[bold purple]🤖 AI Answer:[/bold purple]")
            console.print(Panel(Markdown(response_text), border_style="purple", title="Result", title_align="left"))

            if hasattr(response, 'source_nodes') and response.source_nodes:
                tree = Tree("📚 [dim]Knowledge sources:[/dim]")
                found_sources = False

                for node_score in response.source_nodes:
                    score = node_score.score or 0.0
                    meta = node_score.node.metadata
                    file_name = meta.get('file_name') or meta.get('file_path') or "Unknown"
                    source_branch = tree.add(f"[cyan]{file_name}[/cyan] [dim](Score: {score:.2f})[/dim]")
                    text_preview = node_score.node.get_text().replace('\n', ' ').strip()[:80] + "..."
                    source_branch.add(f"[italic grey50]\"{text_preview}\"[/italic grey50]")
                    found_sources = True

                if found_sources:
                    console.print(tree)
                else:
                    console.print("[dim italic]No sources found (LLM answered from its memory or hallucinated)[/dim]")
                console.print("")
            else:
                 console.print("[dim italic]RAG did not return source nodes.[/dim]\n")
        except KeyboardInterrupt:
            console.print("\n[bold red]⛔ User interruption.[/bold red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]❌ An error occurred:[/bold red] {e}")

if __name__ == "__main__":
    main()
