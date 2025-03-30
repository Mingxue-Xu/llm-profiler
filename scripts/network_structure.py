import torch
from transformers import AutoModelForCausalLM
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn.modules.module import _addindent
import numpy as np


def get_model_structure(model, max_depth=3):
    """Generate a detailed structure of the model up to a certain depth."""
    graph = nx.DiGraph()

    def add_nodes(module, name='', depth=0):
        if depth > max_depth:
            return

        # Add current module as a node
        node_id = name if name else 'model'
        module_type = module.__class__.__name__
        graph.add_node(node_id, type=module_type)

        # Add children modules and connections
        for child_name, child in module.named_children():
            child_id = f"{name}/{child_name}" if name else child_name
            child_type = child.__class__.__name__
            graph.add_node(child_id, type=child_type)
            graph.add_edge(node_id, child_id)

            # Recursively add children
            add_nodes(child, child_id, depth + 1)

    add_nodes(model)
    return graph


def plot_model_structure(graph, figsize=(20, 15)):
    """Plot the model structure using networkx."""
    plt.figure(figsize=figsize)

    # Use hierarchical layout
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

    # Get node types for coloring
    node_types = [graph.nodes[node]['type'] for node in graph.nodes]
    unique_types = list(set(node_types))
    color_map = plt.cm.tab20(np.linspace(0, 1, len(unique_types)))
    type_to_color = {t: color_map[i] for i, t in enumerate(unique_types)}
    node_colors = [type_to_color[graph.nodes[node]['type']] for node in graph.nodes]

    # Draw the graph
    nx.draw(graph, pos, with_labels=True, node_color=node_colors,
            node_size=2000, alpha=0.8, font_size=8, font_weight='bold',
            arrowsize=15, width=1.5)

    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color, markersize=10, label=t)
               for t, color in type_to_color.items()]
    plt.legend(handles=handles, title="Module Types", loc='upper right')

    plt.title("LLM Model Structure", size=20)
    plt.tight_layout()
    plt.savefig("model_structure.png", dpi=300)
    plt.show()


def generate_text_summary(model):
    """Generate a text summary of the model structure."""

    def summarize(module, prefix=''):
        summary = []
        for name, submodule in module.named_children():
            summary.append(f"{prefix}{name}: {submodule.__class__.__name__}")
            child_summary = summarize(submodule, prefix + '  ')
            if child_summary:
                summary.extend(child_summary)
        return summary

    return '\n'.join(summarize(model))


def count_parameters(model):
    """Count the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


# Main function to load and visualize the model
def visualize_llm_structure(model_name="meta-llama/Meta-Llama-3-8B", max_depth=3):
    print(f"Loading model: {model_name}")

    # Load the model (with low memory footprint if possible)
    try:
        # First try loading with 4-bit quantization for memory efficiency
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config
        )
    except:
        # Fall back to standard loading with FP16
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except:
            # Last resort: try loading just the config
            from transformers import AutoConfig
            print("Loading full model failed. Using model config only.")
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_config(config)

    print("Model loaded successfully.")

    # Generate and plot model structure
    print("Generating model structure graph...")
    graph = get_model_structure(model, max_depth=max_depth)

    print(f"Model contains {len(graph.nodes)} modules at depth {max_depth}")
    print(f"Total parameters: {count_parameters(model):,}")

    # Plot the structure
    print("Plotting model structure...")
    plot_model_structure(graph)

    # Generate text summary
    print("\nModel Structure Summary:")
    print(generate_text_summary(model)[:1000] + "...")  # Truncated for brevity

    return model, graph


if __name__ == "__main__":
    # You may need to log in to Hugging Face first with:
    # from huggingface_hub import login
    # login()

    # Visualize the model structure
    model, graph = visualize_llm_structure("meta-llama/Llama-3.2-3B-Instruct", max_depth=6)