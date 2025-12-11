from rich.tree import Tree
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
import sys
import re

from src.rag import (
    get_response,
    check_for_updates,
    update_knowledge_base,
    rebuild_knowledge_base,
    reset_chat_history,
    save_access_control_config,
    get_documents_access_control
)

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

    changes = check_for_updates()
    if changes:
        console.print("\n[bold yellow]📢 Knowledge Base Updates Detected:[/bold yellow]")
        if changes['added']:
            console.print(f"   [green]+ Added: {', '.join(changes['added'])}[/green]")
        if changes['modified']:
            console.print(f"   [blue]~ Modified: {', '.join(changes['modified'])}[/blue]")
        if changes['deleted']:
            console.print(f"   [red]- Deleted: {', '.join(changes['deleted'])}[/red]")

        if Prompt.ask("\n[bold cyan]🔄 Do you want to update the database now?[/bold cyan]", choices=["y", "n"], default="y") == "y":
            with console.status("[bold magenta]🔄 Updating knowledge base...[/bold magenta]"):
                update_knowledge_base(changes)
        else:
             console.print("[dim]Update skipped.[/dim]")

    while True:
        try:
            console.print("\n[bold green]👤 Your question:[/bold green]")
            user_input = Prompt.ask("💬").strip()

            if user_input.lower() in ["exit", "quit", "q", "выход"]:
                console.print("\n[bold yellow]👋 Goodbye! Session terminated.[/bold yellow]")
                break

            if user_input.lower() in ["clear", "cls"]:
                console.clear()
                print_banner()
                continue

            if user_input.lower() in ["rebuild", "rb"]:
                if Prompt.ask("\n[bold red]⚠️ This will rebuild the entire knowledge base. Continue?[/bold red]", choices=["y", "n"], default="n") == "y":
                    with console.status("[bold magenta]🔄 Rebuilding knowledge base...[/bold magenta]"):
                        rebuild_knowledge_base()
                        console.clear()
                        console.print("\n[bold green]✅ Knowledge base rebuilt successfully.[/bold green]")
                else:
                    console.print("[dim]Rebuild cancelled.[/dim]")
                continue

            if user_input.lower() in ["ch", "clear history"]:
                reset_chat_history()
                console.clear()
                console.print("[dim]🧹 Chat history cleared.[/dim]")
                continue

            if user_input.lower() == "admin":
                run_admin_dashboard(console)
                continue

            if user_input.lower() in ["update", "upd"]:
                with console.status("[bold magenta]🔄 Checking for knowledge base updates...[/bold magenta]"):
                    changes = check_for_updates()
                    if changes:
                        console.print("\n[bold yellow]📢 Knowledge Base Updates Detected:[/bold yellow]")
                        if changes['added']:
                            console.print(f"   [green]+ Added: {', '.join(changes['added'])}[/green]")
                        if changes['modified']:
                            console.print(f"   [blue]~ Modified: {', '.join(changes['modified'])}[/blue]")
                        if changes['deleted']:
                            console.print(f"   [red]- Deleted: {', '.join(changes['deleted'])}[/red]")

                        update_knowledge_base(changes)
                    else:
                        console.print("[bold green]✅ Knowledge base is up to date.[/bold green]")
                continue

            if user_input.lower() in ["help", "h", "?"]:
                console.clear()
                help_text = """
                [bold cyan]💡 RAG AI Chat Help:[/bold cyan]
                - Type your question and press Enter to get an answer.
                - To target specific documents, use the syntax: [dim]@filename.ext[/dim] in your question.
                  Example: [dim]What is the summary of the report? @report.pdf[/dim]
                - Type [dim]exit[/dim], [dim]quit[/dim], or [dim]q[/dim] to leave the chat.
                - Type [dim]clear[/dim] or [dim]cls[/dim] to clear the screen.
                - Type [dim]help[/dim], [dim]h[/dim], or [dim]?[/dim] to display this help message.
                - Type [dim]update[/dim] or [dim]upd[/dim] to check for knowledge base updates.
                - Type [dim]rebuild[/dim] or [dim]rb[/dim] to rebuild the entire knowledge base.
                - Type [dim]clear history[/dim] or [dim]ch[/dim] to clear the chat history.
                """
                console.print(Panel(help_text, border_style="cyan", title="Help", title_align="left"))
                continue

            if not user_input.strip():
                continue

            file_filters = re.findall(r'@([\w\.\-_]+)', user_input)
            clean_input = re.sub(r'@[\w\.\-_]+', '', user_input).strip()

            if not clean_input and not file_filters:
                continue

            console.print("")

            if file_filters:
                console.print(f"[dim]🎯 Targeted documents: {', '.join(file_filters)}[/dim]")

            with console.status("[bold magenta]🤖 Reading documents and generating answer...[/bold magenta]", spinner="dots"):
                response = get_response(clean_input, file_filters=file_filters)

            response_text = str(response)

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
                 console.print("[dim italic]RAG did not return source nodes.[/]\n")

            console.print("[bold purple]🤖 AI Answer:[/bold purple]")
            console.print(Panel(Markdown(response_text), border_style="purple", title="Result", title_align="left"))
        except KeyboardInterrupt:
            console.print("\n[bold red]⛔ User interruption.[/bold red]")
            break
        except Exception as e:
            import traceback
            console.print(f"\n[bold red]❌ An error occurred:[/bold red] {escape(str(e))}")
            console.print("[bold red]Traceback:[/bold red]")
            console.print(escape(str(traceback.format_exc())))

def run_admin_dashboard(console: Console):
    files_on_disk, current_config = get_documents_access_control()
    disk_files_set = set(files_on_disk)
    files_to_remove = [f for f in current_config if f not in disk_files_set]

    if files_to_remove:
        console.print(f"[bold yellow]🧹 Cleaning up {len(files_to_remove)} deleted files from config...[/bold yellow]")
        for f in files_to_remove:
            del current_config[f]

    new_files = [f for f in files_on_disk if f not in current_config]

    for f in new_files:
        current_config[f] = "private" # Или 'public', как вам удобнее

    while True:
        console.clear()
        console.print("[bold blue]🛡️  Admin Access Control Panel[/bold blue]")
        console.print("[dim]Existing files missing from config have been added as 'private'.[/dim]")
        console.print("[dim]Files deleted from disk have been removed from config.[/dim]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Filename")
        table.add_column("Access Level", justify="center")

        sorted_files = sorted(files_on_disk)

        for idx, filename in enumerate(sorted_files, 1):
            level = current_config.get(filename, "private")

            if level == "public":
                level_str = "[green]PUBLIC[/green] 🌍"
            else:
                level_str = "[red]PRIVATE[/red] 🔒"

            name_str = filename
            if filename in new_files:
                name_str = f"[bold yellow]{filename} (NEW)[/bold yellow]"

            table.add_row(str(idx), name_str, level_str)

        console.print(table)
        console.print("\n[bold]Options:[/bold]")
        console.print("• Enter [bold cyan]number[/bold cyan] to toggle access (e.g. '1')")
        console.print("• Type [bold green]save[/bold green] to save changes and exit")
        console.print("• Type [bold red]cancel[/bold red] to discard changes")

        choice = Prompt.ask("\nSelect option")

        if choice.lower() == "save":
            save_access_control_config(current_config)
            console.print("[bold green]✅ Config saved successfully![/bold green]")
            break

        elif choice.lower() in ["cancel", "q", "quit", "exit"]:
            console.print("[bold yellow]⚠️  Changes discarded.[/bold yellow]")
            break

        try:
            idx = int(choice)

            if 1 <= idx <= len(sorted_files):
                target_file = sorted_files[idx - 1]
                current_level = current_config.get(target_file, "private")
                new_level = "private" if current_level == "public" else "public"
                current_config[target_file] = new_level
            else:
                console.print("[red]Invalid number![/red]")
        except ValueError:
            pass

if __name__ == "__main__":
    main()
