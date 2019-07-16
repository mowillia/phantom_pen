# Phantom Pen - Web Application 

[This repository is under construction]

These files contain the code for the Flask web application Phantom Pen. The model training (for GPT-2), web-scraping, and data cleaning codes are not included in this repository

![](phantompen_screenshot.png)


### Code Files

The code files have the following functions

* Text Files: Files for various corpora
  * `gutenberg_texts`
  * `short_stories`
  * `Atlantic_subj_essays`
  * `concatenated_subj_files`
* Training Code
  * `gpt2_training_clean.ipynb` Code used to train the various models for the program
* Scraping texts and concatenating
  * `web_scraping_atlantic_subjs.py` Code used to scrape atlantic articles
  * `web_scraping_short_stories.py` Code used to scrape short stories
  * `essay_concatenation_clean.py` Code used to concatenate the articles, short stories, books
* Text generation and Classification
  * `text_generation_function.ipynb` Code for the simple generate, classify & generate, and classify, extract, & generate functions of the application
  * `classification_comparison.ipynb` Comparison of various classifiers for the classify function
* `writrly folder` Code for Flask application
  * HTML/CSS/Image Files 
  * Python File of application
* Additional Code for Flask application
  * needed for sample generation
    * `encoder.py` needed for sample generation
    * `sample.py`
    * `model.py`
  * To run application
    * `run.py`



## Authors

* **Mobolaji Williams** - [mowillia](https://github.com/mowillia)


## Acknowledgments

* [NSheppherd](https://github.com/nshepperd) for code for training
* [OpenAI](https://github.com/openai) for pretrained model
