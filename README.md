# FRIDAY-ML: The AI-Assisted Machine Learning Workbench

![](logo-friday-ml.png)

![Status](https://img.shields.io/badge/Status-Beta-blue) ![Stack](https://img.shields.io/badge/Tech-PyData%20|%20LLMs-purple)

**FRIDAY-ML** is an interactive educational workspace designed to bridge the gap between machine learning theory and practical implementation in Python.

By combining the best data science tools in the PyData ecosystem and the support of [**Claude Code**](https://code.claude.com/docs/en/vs-code) or [**Mistral Vibe**](https://mistral.ai/news/devstral-2-vibe-cli), we allow learners to focus on principles rather than syntax errors.

> [!Tip]
> _Think more, write less, read and evaluate everything._

## Learn how to do machine learing in Python with support from an AI agent

Traditional machine learning (ML) courses often trap students in dependency hell or syntax fatigue, causing them to lose sight of the mathematical and logical principles. FRIDAY-ML flips this model:

1.  **AI writes most of the Python code:** An AI agent helps you write the machine learning code in Python.
2.  **You learn how to think like a data scientist:** Using [Instroduction to Statistical Learning](https://www.statlearning.com/) as our stepping stone, your AI-assistant F.R.I.D.A.Y. challenges you to dictate without giving away the answer too much.
    - **Conceptualize:** Think about what you want to do (_"I need to reduce variance in this model"_).
    - **Prompt:** Instruct the AI agent to implement a specific validation strategy (_"Set up a K-Fold cross-validation loop using scikit-learn"_).
    - **Audit:** Read the Python code. Does the code actually do what you asked? Are the data transformations correct?
    - **Execute:** Run the reactive cell and analyze the visualization.
3.  **You learn how to review the code:** Your primary job is to read the generated Python, understand the data flow, and evaluate the results. The aim is to become proficient in reading Python code. As with natural languages, it is easier to obtain passive, working knowledge.

## Our stack

* **[Positron](https://positron.posit.co/):** the best open source data science IDE (integrated development environemnt) that unifies exploratory data analysis, machine learning and a fully integrated AI assistant.
* **[marimo](https://github.com/marimo-team/marimo):** a next-generation reactive notebook for Python. No more hidden state or out-of-order execution errors.
* **[scikit-learn](https://scikit-learn.org/):** the standard library for tabular machine learning in Python - linear models, tree-based models, clustering, and model evaluation.

<details>
<summary>Why we use Positron</summary>

While Positron is actually built on the same foundation as VS Code (Code OSS), it removes the "assembly required" aspect of setting up a data science environment. If VS Code is a box of Lego bricks, Positron is the pre-built model designed specifically for R and Python.

Positron comes with specialized data panes, similar to those in RStudio.

- **Data Explorer:** a built-in spreadsheet viewer that handles millions of rows without lagging. You can filter, sort, and search dataframes without writing a single line of df.head(). Open `.parquet` files directly by just double-clicking.

- **Variables Pane:** a real-time view of your environment (objects, types, and values) that is much more intuitive than the standard "Variables" tab in the VS Code debugger.

- **Plots & Viewer:** Dedicated spaces for visualizations and HTML widgets (like Leaflet or Shiny apps) that don't get lost in your editor tabs.
</details>

<details>
<summary>Why we use marimo</summary>

Traditional notebooks such as Jupyter notebooks as well as commercial notebooks such as Deepnote and Hex are ill-suited for use with agentic coding tools.

- **File format.** By default Jupyter notebooks are stored as JSON with base64-encoded outputs, not Python. But LLMs work best when generating code, and marimo is stored as Python, not JSON, empowering agents to do their best work.

- **Reproducibility.** Jupyter notebooks, as well as their commercial skins, suffer from a reproducibility crisis: they are not well-formed programs but instead have hidden state and hidden bugs. This trips up not only humans but also agents. In contrast, marimo notebooks are reproducible Python programs with well-defined execution semantics, backed by a dataflow graph.

- **Composability and programmability.** Commercial notebooks like Hex and Deepnote provide a point-and-click UI for creating interactive elements such as sliders, text boxes, dropdowns, and more. LLMs-based agents struggle with this; in contrast, marimo is code-first, letting humans and agents alike create powerful UIs with just Python.

(Of course, commercial notebooks are proprietary, so you couldn’t author them locally from your terminal even if you wanted to. Because marimo is open-source, you can use it locally, on cloud servers, wherever you like.)

- **Introspection.** Agents work best when they can test what they’ve done. marimo notebooks are Python programs, so agents can run them and inspect outputs; the same is not true for Jupyter notebooks.

For more details, see [this blogpost](https://marimo.io/blog/claude-code)
</details>

## Getting started

### Install Positron

1. Download Positron from the [official website](https://positron.posit.co/)
2. Install the application for your operating system (macOS, Windows, or Linux)
3. Launch Positron

### Add the marimo extension

1. Click on the Extensions icon on the left-sidebar
2. Search for the [marimo](https://open-vsx.org/extension/marimo-team/vscode-marimo) and install the extension


### Install FRIDAY-ML

1. Download FRIDAY-ML [here](https://github.com/EAISI/FRIDAY-ML/archive/refs/heads/main.zip)
2. Unzip the files and move the whole folder to working directory
3. Go back to Positron and add the folder to the workspace via `File > Add Folder to Workspace...`

### Install uv (if not already installed)

1. Open a terminal in Positron via `Terminal > New Terminal`
2. Run the following command in the terminal:

    - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
    ```

###  Install Python dependencies

In the same terminal, install the Python dependencies for FRIDAY-ML

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


## Using FRIDAY-ML with Positron Assistant

Once you have installed all the software components you can start using FRIDAY-ML with Positron Assistant.

1. Open the FRIDAY-ML folder in Positron
2. The Python interpreter should automatically use the `.venv` created by `uv`
3. Open any `.py` marimo notebook file (e.g., `notebooks/ames-housing.py`)
4. The integrated marimo extension allows you to run and edit notebooks directly in Positron
5. Use Positron Assistant to invoke `/friday` (more details below)
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
> 3. The notebook will open in an integrated panel within Positron, showing all cells and outputs
>
> Alternatively, you can right-click on any `.py` marimo notebook file and select **"Open with Marimo"** from the context menu.


### Using the `/friday` prompt

The FRIDAY skill is an AI tutor built into this workspace, designed to coach you through machine learning concepts without getting bogged down in Python syntax. It helps you understand ML principles by:

- Explaining concepts in plain language
- Breaking down complex algorithms into digestible steps
- Answering "why" questions about model behavior
- Guiding you through theoretical foundations
- Connecting mathematical concepts to practical implementation

You can invoke it by typing `/friday` in Positron assistant or include it in your prompts:
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
We salute the creators of the ['real' F.R.I.D.A.Y.](https://marvelcinematicuniverse.fandom.com/wiki/F.R.I.D.A.Y.), with a nerdy wink of an eye. and also because most lectures as [EAISI Academy](https://www.tue.nl/en/education/professional-education/current-programs/eaisi-academy) are held on Fridays. This project is licensed under the MIT License.
