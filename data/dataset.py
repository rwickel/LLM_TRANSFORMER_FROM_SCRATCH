from datasets import load_dataset

# Load TriviaQA dataset
def load_triviaqa(sample_size=None):
    dataset = load_dataset("trivia_qa", "unfiltered")
    if sample_size:
        dataset["train"] = dataset["train"].select(range(sample_size))
    return dataset

# Load SQuAD dataset
def load_squad_dataset(sample_size=None, name="squad"):
    squad_dataset = load_dataset(name)
    qa_dataset = Dataset(source=name)
    data = squad_dataset['train'] if sample_size is None else squad_dataset['train'].select(range(sample_size))

    def is_single_word_answer(example):
        return len(example['answers']['text'][0].split()) == 1

    data = data.filter(is_single_word_answer)

    for item in data:
        query = item['question']
        context = item['context']
        answer = item['answers']['text'][0] if item['answers']['text'] else "No answer"
        qa_instance = QuestionAnswering(query=query, answer=answer, text=context)
        qa_dataset.list.append(qa_instance)

    return qa_dataset

def load_cbt_dataset(sample_size=None, name="cbt", config="NE"):
    """
    Loads the Children's Book Test dataset and populates a Dataset instance with QuestionAnswering objects.
    
    Args:
        sample_size (int or None): The number of samples to load. If None, loads the entire dataset.
        name (str): The name of the dataset to load.
        config (str): The configuration of the dataset (e.g., 'NE' for English).
    
    Returns:
        Dataset: A Dataset instance containing QuestionAnswering objects.
    """
    # Load dataset with specified configuration
    cbt_dataset = load_dataset(name, config)

    # Create a Dataset instance with source info
    qa_dataset = Dataset(source=f"{name} {config}")

    # Select the whole dataset if sample_size is None
    data = cbt_dataset['train'] if sample_size is None else cbt_dataset['train'].select(range(sample_size))

    # Populate Dataset with QuestionAnswering instances
    for item in data:
        context = item['sentences']  # This is the passage or context in CBT
        query = item['question']  # The question related to the passage
        answer = item['answer']  # The answer to the question
        
        # If no answer exists, default to "No answer"
        answer = answer if answer else "No answer"
        
        # Create QuestionAnswering instance
        qa_instance = QuestionAnswering(query=query, answer=answer, text=context)
        qa_dataset.list.append(qa_instance)
    
    return qa_dataset



def load_marco_dataset(sample_size=None, name="ms_marco", version="v2.1"):
    """
    Loads a specified version of the MS MARCO dataset and populates a Dataset instance with QuestionAnswering objects.

    Args:
        sample_size (int or None): The number of samples to load. If None, loads the entire dataset.
        name (str): The name of the dataset to load.
        version (str): The version of the dataset to load.

    Returns:
        Dataset: A Dataset instance containing QuestionAnswering objects.
    """
    # Load dataset with specified name and version
    marco_dataset = load_dataset(name, version)

    # Create a Dataset instance with source info
    qa_dataset = Dataset(source=f"{name} {version}")

    # Select the whole dataset if sample_size is None
    data = marco_dataset['train'] if sample_size is None else marco_dataset['train'].select(range(sample_size))

    # Populate Dataset with QuestionAnswering instances
    for item in data:
        query = item['query']
        answers = item['answers']
        passages =item['passages']['passage_text']
        # Get the first available answer or a default "No answer"
        answer = answers[0] if answers else "No answer"
        
        qa_instance = QuestionAnswering(query=query, answer=answer, text=passages, relevant=item['passages']['is_selected'])
        qa_dataset.list.append(qa_instance)
    
    return qa_dataset
