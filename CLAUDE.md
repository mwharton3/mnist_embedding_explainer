I'm creating an explainer document to help communicate the concept of embeddings.

I want you to do a few things:
- Pull a local copy of the mnist digits dataset into data/
- Create a reusable training pipeline that predicts labels assigned at train-time
- Use a basic CNN for the architecture with sensible defaults
- For the training tasks, train two models: one that predicts the digit, and another that predicts even vs. odd. 
    - To be specific, "even" means the digit is 0, 2, 4, 6, or 8, and "odd" means the digit is 1, 3, 5, 7, or 9
- Train on the 60k train samples, but ignore the 10k test samples in the train loop
    - Search the web for good hyperparameters to start with
    - No need to get fancy: you can use a static learning rate and patience (10 is a good starting point) for stopping criteria
- Once you've trained these models, save off the weights so they are reusable
- Subsample 1000 random data points from the test set
- With each of the two models:
    - Embed the 1000 samples (extract the penultimate layer in the FC head)
    - Project the embeddings to 2D using umap
    - Generate a scatterplot that colors everything black, except for the digit "7"
- Generate two more exprimental plots where you overlay some sample images onto the previous plots, i.e. an actual thumbnail of a specific digit with a line segment linking to the datapoint itself. I want to show two examples that are "similar" (both digit 7) in the plot that predicts the digit itself. I want two more examples in the second plot where the digits are DIFFERENT (though one is 7 and the other is odd), which shows how similarity is up to you to define.

Requirements:
- uv for package manager
- python for code
- use umap for projection of vectors on to 2D plot 
- pytorch for the framework
- matplotlib for plots
- I want to tune the plotting script, keep that separate and configurable with helpful docstrings
- Store all your code in src/
- Create a docs/ folder and at minimum, include METHODOLOGY.md (and any other helpful documentation you decide to include)
- Do your best to make the plot beautiful. I'm posting on LinkedIn, so it must be visually appealing.
- Use git along the way for feature-level commits, but only commit locally. I want full traceability of what you write vs. what I write.

Let me know if I can provide any clarity along the way.
