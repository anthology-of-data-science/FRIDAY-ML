# FRIDAY-ML: The AI-Assisted Machine Learning Workbench

![](logo-friday-ml.png)

![Status](https://img.shields.io/badge/Status-Beta-blue) ![Stack](https://img.shields.io/badge/Tech-PyData%20|%20LLMs-purple)

**FRIDAY-ML** is an interactive educational workspace designed to bridge the gap between machine learning theory and practical implementation in Python.

By combining the best data science tools in the PyData ecosystem and the support of [**Claude Code**](https://code.claude.com/docs/en/vs-code) or [**Mistral Vibe**](https://mistral.ai/news/devstral-2-vibe-cli), we allow learners to focus on principles rather than syntax errors.

> [!Tip]
> _Think more, write less, read and evaluate everything._

## Learn how to do machine learing in Python with support from an AI agent

Traditional machine learning (ML) courses often trap students in _dependency hell_ or _syntax fatigue_, causing them to lose sight of the mathematical and logical principles. FRIDAY-ML flips this model:

1.  **AI writes most of the Python code:** An AI agent helps you write the machine learning code in Python.
2.  **You learn how to think like a data scientist:** Using [Instroduction to Statistical Learning](https://www.statlearning.com/) as our stepping stone, your AI-assistant F.R.I.D.A.Y. challenges you to dictate without giving away the answer too much.
    - **Conceptualize:** Think about what you want to do (_"I need to reduce variance in this model"_).
    - **Prompt:** Instruct the AI agent to implement a specific validation strategy (_"Set up a K-Fold cross-validation loop using scikit-learn"_).
    - **Audit:** Read the Python code. Does the code actually do what you asked? Are the data transformations correct?
    - **Execute:** Run the reactive cell and analyze the visualization.
3.  **You learn how to review the code:** Your primary job is to read the generated Python, understand the data flow, and evaluate the results. The aim is to become proficient in reading Python code. As with natural languages, it is easier to obtain passive, working knowledge.

## Our stack

* **[Marimo](https://github.com/marimo-team/marimo):** a next-generation reactive notebook for Python. No more hidden state or out-of-order execution errors.
* **[Positron](https://positron.posit.co/):** the best open source data science IDE (integrated development environemnt) that unifies exploratory data analysis and production work.
* **[scikit-learn](https://scikit-learn.org/):** the standard library for tabular machine learning in Python - linear models, tree-based models, clustering, and model evaluation.
* **[AutoGluon](https://auto.gluon.ai/):** an AutoML library to help you with experimentation and try many models in one go.
* **AI Coding Agents:** Integration with [**Anthropic's Claude Code**](https://code.claude.com/docs/en/vs-code) or [**Mistral's Vibe with Devstral**](https://mistral.ai/news/devstral-2-vibe-cli) to act as your pair programmer assistant.

<details>
<summary>Why we use Marimo</summary>

Traditional notebooks such as Jupyter notebooks as well as commercial notebooks such as Deepnote and Hex are ill-suited for use with agentic coding tools.

- **File format.** By default Jupyter notebooks are stored as JSON with base64-encoded outputs, not Python. But LLMs work best when generating code, and marimo is stored as Python, not JSON, empowering agents to do their best work.

- **Reproducibility.** Jupyter notebooks, as well as their commercial skins, suffer from a reproducibility crisis: they are not well-formed programs but instead have hidden state and hidden bugs. This trips up not only humans but also agents. In contrast, marimo notebooks are reproducible Python programs with well-defined execution semantics, backed by a dataflow graph.

- **Composability and programmability.** Commercial notebooks like Hex and Deepnote provide a point-and-click UI for creating interactive elements such as sliders, text boxes, dropdowns, and more. LLMs-based agents struggle with this; in contrast, marimo is code-first, letting humans and agents alike create powerful UIs with just Python.

(Of course, commercial notebooks are proprietary, so you couldn’t author them locally from your terminal even if you wanted to. Because marimo is open-source, you can use it locally, on cloud servers, wherever you like.)

- **Introspection.** Agents work best when they can test what they’ve done. marimo notebooks are Python programs, so agents can run them and inspect outputs; the same is not true for Jupyter notebooks.

For more details, see [this blogpost](https://marimo.io/blog/claude-code)
</details>

## Getting started

> [!NOTE]
> 
> **Prerequisites**
> * Python 3.12+
> * An account and API key for Anthropic Claude Code or Mistral Vibe Devstral.

<details>
<summary><b>Install Positron</b></summary>

<br>[Positron](https://positron.posit.co/) is an open-source data science IDE built on VS Code, designed specifically for data science workflows with enhanced support for Python, R, and interactive notebooks.

### Installing Positron

1. Download Positron from the [official website](https://positron.posit.co/)
2. Install the application for your operating system (macOS, Windows, or Linux)
3. Launch Positron

### Importing the FRIDAY-ML Code Profile

This repository includes a pre-configured code profile ([friday-ml.code-profile](friday-ml.code-profile)) with recommended settings and extensions for the best learning experience.

To import the profile:

1. Open Positron
2. Go to **File > Preferences > Profiles > Import Profile...**
3. Select the `friday-ml.code-profile` file from the repository root
4. Click **Create Profile** to complete the import

The profile includes:
- Optimized theme and editor settings for data science
- Essential extensions: Claude Code, Ruff, Marimo, and more
- Pre-configured Python interpreter settings
- Integrated viewer for web-based visualizations

</details>

<details>
<summary><b>Install FRIDAY-ML</b></summary>
    <br>
    1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/FRIDAY-ML.git
    cd FRIDAY-ML
    ```

2.  **Install uv (if not already installed):**

    macOS/Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Windows:
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

3.  **Install dependencies:**

    For standard installation:
    ```bash
    uv sync
    ```

    For Apple Silicon with TensorFlow support:
    ```bash
    uv sync --extra tf-apple
    ```

    For other systems with TensorFlow support:
    ```bash
    uv sync --extra tf
    ```

4.  **Set Environment Variables:**
    Create a `.env` file and add your API keys:
    ```bash
    ANTHROPIC_API_KEY=sk-ant-...
    # or
    MISTRAL_API_KEY=...
    ```
</details>




   

## Using FRIDAY-ML

There are two options to work with marimo notebooks in this workspace:

### Option 1: Positron with Marimo Extension (Recommended)

This is the default and most integrated approach for data science workflows. You benefit from the unified Integrated Development Environment (IDE) experience for editing and running notebooks. Positron also has a [data explorer](https://positron.posit.co/data-explorer.html) function to browse `.csv` and `.parquet` files. To use this workflow:

1. Open the FRIDAY-ML folder in Positron
2. The Python interpreter should automatically use the `.venv` created by `uv`
3. Open any `.py` marimo notebook file (e.g., `notebooks/mnist.py`)
4. The integrated Marimo extension allows you to run and edit notebooks directly in Positron
5. Use Claude Code in the panel or sidebar for AI-assisted development
6. All notebook outputs and visualizations appear inline in the editor

> [!TIP]
>
> You can open a marimo notebook (`.py` file in the `notebooks/` directory) using the UI: 
>
> 1. **Navigate to the notebook file** in the Positron file explorer (e.g., `notebooks/mnist.py`)
> 2. **Click the marimo icon** in the top-right corner of the editor:
>
>   ![Open Marimo Icon](open-marimo.png)
> 
>3. The notebook will open in an integrated panel within Positron, showing all cells and outputs
>
> Alternatively, you can right-click on any `.py` marimo notebook file and select **"Open with Marimo"** from the context menu.



### Option 2: Marimo UI with Watch Mode

This option uses marimo's native web interface with automatic reloading, directly from the terminal. It gives you full marimo interactive features in the browser, which is more easy going than Positron.
This workflow is described in the [marimo blog post](https://marimo.io/blog/claude-code). To use it, do the following:

1. **Terminal 1 - Start marimo in watch mode:**
   ```bash
   uv run marimo edit <notebook>.py --watch
   ```

   For example:
   ```bash
   uv run marimo edit notebooks/mnist.py --watch
   ```
   
   This opens the marimo notebook in your browser and watches for file changes.

2. **Terminal 2 - Launch Claude Code:**
   ```bash
   claude
   ```

   Use Claude to edit the notebook file. The `--watch` flag ensures marimo automatically reloads changes in the browser.

### Viewing tensorboard training logs (only for `mnist.py` notebook)

To view TensorBoard logs from your training runs:

```bash
uv run tensorboard --logdir logs/
```

## Using the FRIDAY Skill

The FRIDAY skill is a specialized AI tutor built into this workspace, designed to coach you through machine learning concepts without getting bogged down in Python syntax.

### What is the FRIDAY Skill?

The FRIDAY skill helps you understand ML principles by:
- Explaining concepts in plain language
- Breaking down complex algorithms into digestible steps
- Answering "why" questions about model behavior
- Guiding you through theoretical foundations
- Connecting mathematical concepts to practical implementation

### How to use it

When working with Claude Code or Vibe, invoke the FRIDAY skill by typing:

```bash
/friday
```

or include it in your prompts:
```bash
/friday Why does my model have high variance?
/friday Explain the bias-variance tradeoff in this context
/friday What's the intuition behind gradient descent?
/friday Help me understand why cross-validation matters
```

### When to use FRIDAY vs. direct code generation

- **Use `/friday`** when you want to understand the "why" behind a concept or need conceptual guidance before implementation
- **Use direct prompts** when you want the AI agent to generate or modify code

> [!TIP]
>
> #### Example Workflow
>
> 1. **Start with a conceptual question:**
>   ```bash
>   /friday Why would I use L2 regularization instead of L1 for this regression problem?
>   ```
>
> 2. **FRIDAY explains the concept** in plain language, focusing on principles rather than syntax
>
> 3. **Once you understand, prompt for implementation:**
>   ```bash
>   Now implement Ridge regression with 5-fold cross-validation on the housing dataset
>   ```
>
>4. **Audit the generated code** with your new understanding of the underlying principles
>
> This separation between learning and coding maintains the "Write less, read more, evaluate everything" philosophy by ensuring you understand concepts before generating implementation code.

## Attribution & license 
We salute the creators of the ['real' F.R.I.D.A.Y.](https://marvelcinematicuniverse.fandom.com/wiki/F.R.I.D.A.Y.), with a nerdy wink of an eye. This project is licensed under the MIT License.
