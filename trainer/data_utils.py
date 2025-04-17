# trainer/data_utils.py
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import os
import re
import torch
import pyarrow
from pathlib import Path

# Import the config class
from .config import TrainingConfig

# --- Helper Preparation Functions ---
# Define these before load_and_prepare_data or import them

def prepare_trivia_qa(examples, tokenizer, max_context_snippets=3, context_separator="\n"):
    """
    Combines context (from search_results['search_context']), question, and answer
    from trivia_qa into 'text_to_tokenize'.
    CORRECTED AGAIN based on the LATEST provided search_results structure.
    """
    eos = tokenizer.eos_token if tokenizer.eos_token else ""
    texts = []

    # Ensure we iterate through the necessary fields
    # Assuming examples['search_results'] provides the dict shown for each item in the batch
    search_results_list = examples.get('search_results', []) # Get the list for the batch

    # Check lengths match, handle potential errors
    if not (len(examples['question']) == len(examples['answer']) == len(search_results_list)):
        print("Warning: Length mismatch between question/answer/search_results in batch. Skipping context for safety.")
        # Fallback: Process without context if lengths mismatch
        for q, a in zip(examples['question'], examples['answer']):
            answer_value = a.get('value', '') if isinstance(a, dict) else str(a)
            texts.append(f"Question: {q}\n{answer_value}{eos}")
        examples["text_to_tokenize"] = texts
        return examples

    # Process normally if lengths match
    for q, a, sr in zip(examples['question'], examples['answer'], search_results_list):
        answer_value = a.get('value', '') if isinstance(a, dict) else str(a)

        # --- Extract and Combine Context Snippets ---
        context_snippets = []
        # NOW: sr is the dictionary containing lists like 'search_context', 'description'
        if isinstance(sr, dict):
            # Get the list of context strings directly from the key
            context_list = sr.get('search_context') # Use 'search_context' as primary source

            if context_list and isinstance(context_list, list):
                num_snippets_to_use = min(len(context_list), max_context_snippets)
                for i in range(num_snippets_to_use):
                    snippet = context_list[i]
                    # Ensure the snippet is a non-empty string
                    if snippet and isinstance(snippet, str):
                        context_snippets.append(snippet.strip())

        # Join the collected snippets
        context_str = context_separator.join(context_snippets)
        # --- End Context Extraction ---

        # --- Format the final string ---
        if context_str:
            texts.append(f"{context_str}{context_separator}{q}{answer_value}{eos}")
        else:
            # Fallback if no context snippets were found for THIS example
            texts.append(f"Question: {q}\n{answer_value}{eos}")

    examples["text_to_tokenize"] = texts
    return examples


def prepare_simple_text_dataset(examples, tokenizer=None):
    """Assumes input has 'text', copies/renames it to 'text_to_tokenize'."""
    if "text" in examples:
        # Ensure text is string and handle potential None values robustly
        texts = [str(t) if t is not None else "" for t in examples["text"]]
        examples["text_to_tokenize"] = texts
    else:
        print(f"Warning: Expected 'text' column not found during simple preparation.")
        # Determine batch size to create appropriate number of empty strings
        # Find a column that exists in the examples to get the length
        example_key = next(iter(examples), None)
        batch_size = len(examples[example_key]) if example_key else 0
        examples["text_to_tokenize"] = [""] * batch_size
    return examples

def save_tokenized_dataset(dataset, save_dir, dataset_name, split_name):
    """Saves a tokenized dataset split to disk."""

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{dataset_name}_{split_name}.pt")
    print(f"Saving tokenized {split_name} to: {save_path}")
    torch.save(dataset, save_path)


def load_tokenized_dataset(save_dir, dataset_name, split_name):
    """Loads a tokenized dataset split from disk."""

    load_path = os.path.join(save_dir, f"{dataset_name}_{split_name}.pt")
    if os.path.exists(load_path):
        print(f"Loading tokenized {split_name} from: {load_path}")
        return torch.load(load_path)
    else:
        return None



def tokenize_function(examples: dict, tokenizer, block_size: int) -> dict:
    """
    Tokenizes a batch of text examples and prepares them for model input.

    Args:
        examples: A dictionary where keys are column names and values are lists
                  of corresponding data (e.g., {"text_to_tokenize": ["text1", "text2", ...]})
        tokenizer: A PreTrainedTokenizerFast instance from the Transformers library.
        block_size: The maximum sequence length for truncation/padding.

    Returns:
        A dictionary containing tokenized input features (input_ids, attention_mask, etc.)
        as PyTorch tensors.
    """

    output = tokenizer(
        examples["text_to_tokenize"],
        truncation=True,
        max_length=block_size,
        padding=False,  # Crucial: Handle padding later in DataLoader
        return_tensors="pt",
    )

    # Clean up and ensure correct shape/dtype
    output["input_ids"] = output["input_ids"].squeeze(1).long()  # Remove batch dim, ensure long
    output["attention_mask"] = output["attention_mask"].squeeze(1).int()  # Remove batch dim, int
    if "token_type_ids" in output:
        output["token_type_ids"] = output["token_type_ids"].squeeze(1).int()  # Remove batch dim, int

    return output

def prepare_trivia_qa(examples, tokenizer, max_context_snippets=3, context_separator="\n"):
    """
    Combines context (from search_results['search_context']), question, and answer
    from trivia_qa into 'text_to_tokenize'.
    CORRECTED AGAIN based on the LATEST provided search_results structure.
    """
    eos = tokenizer.eos_token if tokenizer.eos_token else ""
    texts = []

    # Ensure we iterate through the necessary fields
    # Assuming examples['search_results'] provides the dict shown for each item in the batch
    search_results_list = examples.get('search_results', []) # Get the list for the batch

    # Check lengths match, handle potential errors
    if not (len(examples['question']) == len(examples['answer']) == len(search_results_list)):
        print("Warning: Length mismatch between question/answer/search_results in batch. Skipping context for safety.")
        # Fallback: Process without context if lengths mismatch
        for q, a in zip(examples['question'], examples['answer']):
            answer_value = a.get('value', '') if isinstance(a, dict) else str(a)
            texts.append(f"Question: {q}\n{answer_value}{eos}")
        examples["text_to_tokenize"] = texts
        return examples

    # Process normally if lengths match
    for q, a, sr in zip(examples['question'], examples['answer'], search_results_list):
        answer_value = a.get('value', '') if isinstance(a, dict) else str(a)

        # --- Extract and Combine Context Snippets ---
        context_snippets = []
        # NOW: sr is the dictionary containing lists like 'search_context', 'description'
        if isinstance(sr, dict):
            # Get the list of context strings directly from the key
            context_list = sr.get('search_context') # Use 'search_context' as primary source

            if context_list and isinstance(context_list, list):
                num_snippets_to_use = min(len(context_list), max_context_snippets)
                for i in range(num_snippets_to_use):
                    snippet = context_list[i]
                    # Ensure the snippet is a non-empty string
                    if snippet and isinstance(snippet, str):
                        context_snippets.append(snippet.strip())

        # Join the collected snippets
        context_str = context_separator.join(context_snippets)
        # --- End Context Extraction ---

        # --- Format the final string ---
        if context_str:
            texts.append(f"{context_str}{context_separator}{q}{answer_value}{eos}")
        else:
            # Fallback if no context snippets were found for THIS example
            texts.append(f"Question: {q}\n{answer_value}{eos}")

    examples["text_to_tokenize"] = texts
    return examples

def default_qa(tokenizer):
    """
    Creates a small, kid-friendly QA dataset with prewritten context, questions, and answers.
    Returns a 'text_to_tokenize' column.
    """
    eos = tokenizer.eos_token if tokenizer.eos_token else ""
    
    texts = [
        f"The rabbit hopped through the green field.\nWhere did the rabbit hop?\nThrough the green field.{eos}",
        f"The rabbit hopped over the trunk.\nWhere did the rabbit hop?\nOver the trunk.{eos}",
        f"The cat slept on the warm couch.\nWhere did the cat sleep?\nOn the warm couch.{eos}",
        f"The dog chased the red ball.\nWhat did the dog chase?\nThe red ball.{eos}",
        f"The bird sang on the tree.\nWhere did the bird sing?\nOn the tree.{eos}",
        f"Tom rode his bike to school.\nWhere did Tom ride his bike?\nTo school.{eos}",
        f"The fish swam in the pond.\nWhere did the fish swim?\nIn the pond.{eos}",
        f"Emma ate a sweet cupcake.\nWhat did Emma eat?\nA sweet cupcake.{eos}",
        f"The butterfly flew over the flowers.\nWhat did the butterfly fly over?\nThe flowers.{eos}",
        f"The teddy bear sat on the bed.\nWhere was the teddy bear?\nOn the bed.{eos}",
        f"Lucy picked a red apple from the tree.\nWhat did Lucy pick?\nA red apple.{eos}",
        f"Johnny kicked the soccer ball.\nWhat did Johnny kick?\nThe soccer ball.{eos}",
        f"Max climbed the tall tree.\nWhat did Max climb?\nThe tall tree.{eos}",
        f"The dog barked loudly at the door.\nWhat did the dog bark at?\nThe door.{eos}",
        f"The boy rode his skateboard down the hill.\nWhat did the boy ride?\nHis skateboard.{eos}",
        f"Sarah made a big sandwich.\nWhat did Sarah make?\nA big sandwich.{eos}",
        f"The baby smiled at the teddy bear.\nWho did the baby smile at?\nThe teddy bear.{eos}",
        f"The boy opened his birthday gift.\nWhat did the boy open?\nHis birthday gift.{eos}",
        f"The cat chased the ball of yarn.\nWhat did the cat chase?\nThe ball of yarn.{eos}",
        f"Tommy wore a blue hat.\nWhat did Tommy wear?\nA blue hat.{eos}",
        f"The flowers bloomed in the garden.\nWhere did the flowers bloom?\nIn the garden.{eos}",
        f"The moon shone brightly at night.\nWhen did the moon shine?\nAt night.{eos}",
        f"The stars twinkled in the sky.\nWhere did the stars twinkle?\nIn the sky.{eos}",
        f"The truck drove down the road.\nWhere did the truck drive?\nDown the road.{eos}",
        f"A bird flew in the sky.\nWhere did the bird fly?\nIn the sky.{eos}",
        f"The bear walked through the forest.\nWhere did the bear walk?\nThrough the forest.{eos}",
        f"Sarah painted a picture of the sun.\nWhat did Sarah paint?\nA picture of the sun.{eos}",
        f"The dog slept in its bed.\nWhere did the dog sleep?\nIn its bed.{eos}",
        f"The rabbit nibbled on a carrot.\nWhat did the rabbit nibble on?\nA carrot.{eos}",
        f"The horse ran across the field.\nWhere did the horse run?\nAcross the field.{eos}",
        f"Anna jumped in the puddle.\nWhat did Anna jump in?\nThe puddle.{eos}",
        f"The clock ticked loudly.\nWhat ticked loudly?\nThe clock.{eos}",
        f"Jack built a sandcastle on the beach.\nWhere did Jack build a sandcastle?\nOn the beach.{eos}",
        f"The squirrel hid an acorn under the tree.\nWhere did the squirrel hide an acorn?\nUnder the tree.{eos}",
        f"The girl drank a glass of milk.\nWhat did the girl drink?\nA glass of milk.{eos}",
        f"Ben threw a stone into the lake.\nWhat did Ben throw?\nA stone.{eos}",
        f"The baby crawled across the floor.\nWhere did the baby crawl?\nAcross the floor.{eos}",
        f"Lily read a storybook before bed.\nWhat did Lily read?\nA storybook.{eos}",
        f"The wind blew through the trees.\nWhat blew through the trees?\nThe wind.{eos}",
        f"Alex found a shiny coin on the ground.\nWhat did Alex find?\nA shiny coin.{eos}",
        f"The kids played tag in the park.\nWhere did the kids play tag?\nIn the park.{eos}",
        f"Jenny painted her nails pink.\nWhat did Jenny paint?\nHer nails.{eos}",
        f"The frog jumped into the pond.\nWhere did the frog jump?\nInto the pond.{eos}",
        f"The sun rose behind the mountains.\nWhere did the sun rise?\nBehind the mountains.{eos}",
        f"David planted flowers in the backyard.\nWhere did David plant flowers?\nIn the backyard.{eos}",
        f"The owl hooted in the night.\nWhen did the owl hoot?\nIn the night.{eos}",
        f"Mia drew a heart on the paper.\nWhat did Mia draw?\nA heart.{eos}",
        f"The kitten climbed onto the windowsill.\nWhere did the kitten climb?\nOnto the windowsill.{eos}",
        f"Noah threw the ball over the fence.\nWhere did Noah throw the ball?\nOver the fence.{eos}",
        f"Sammy found a shell at the beach.\nWhat did Sammy find?\nA shell.{eos}",
        f"The leaves fell from the tree.\nWhat fell from the tree?\nThe leaves.{eos}",
        f"The boy whispered to his friend.\nWho did the boy whisper to?\nHis friend.{eos}",
        f"Lucy danced in the rain.\nWhere did Lucy dance?\nIn the rain.{eos}",
        f"The candle flickered in the dark room.\nWhere did the candle flicker?\nIn the dark room.{eos}",
        f"Mark ate a slice of pizza.\nWhat did Mark eat?\nA slice of pizza.{eos}",
        f"The children laughed at the clown.\nWho did the children laugh at?\nThe clown.{eos}",
        f"The rain fell on the roof.\nWhere did the rain fall?\nOn the roof.{eos}",
        f"Olivia watched a movie with her brother.\nWhat did Olivia watch?\nA movie.{eos}",
        f"The goat chewed on some grass.\nWhat did the goat chew on?\nSome grass.{eos}",
        f"James wrote a letter to his friend.\nWhat did James write?\nA letter.{eos}",
        f"The plane flew above the clouds.\nWhere did the plane fly?\nAbove the clouds.{eos}",
        f"The children built a fort with blankets.\nWhat did the children build?\nA fort.{eos}",
        f"Sophia listened to music in her room.\nWhat did Sophia listen to?\nMusic.{eos}",
        f"The cat hid under the table.\nWhere did the cat hide?\nUnder the table.{eos}",
        f"The ice melted in the sun.\nWhat melted in the sun?\nThe ice.{eos}",
        f"Liam fed the ducks at the pond.\nWhat did Liam feed?\nThe ducks.{eos}",
        f"The boy dropped his toy on the floor.\nWhat did the boy drop?\nHis toy.{eos}",
        f"Zoe tied her shoes before running.\nWhat did Zoe tie?\nHer shoes.{eos}",
        f"The spider spun a web in the corner.\nWhere did the spider spin a web?\nIn the corner.{eos}",
        f"Emma opened the window.\nWhat did Emma open?\nThe window.{eos}",
        f"The kangaroo jumped across the path.\nWhere did the kangaroo jump?\nAcross the path.{eos}",
        f"Leo colored the picture with crayons.\nWhat did Leo color?\nThe picture.{eos}",
        f"The girl smiled at the camera.\nWhat did the girl smile at?\nThe camera.{eos}",
        f"The puppy wagged its tail.\nWhat did the puppy wag?\nIts tail.{eos}",
        f"The cat chased the ball of yarn.\nWhat did the cat chase?\nThe ball of yarn.{eos}",
        f"Tommy wore a blue hat.\nWhat did Tommy wear?\nA blue hat.{eos}",
        f"Sarah painted a picture of the sun.\nWhat did Sarah paint?\nA picture of the sun.{eos}",
        f"David planted flowers in the backyard.\nWhat did David plant?\nFlowers.{eos}",
        f"Mia drew a heart on the paper.\nWhat did Mia draw?\nA heart.{eos}",
        f"The boy dropped his toy on the floor.\nWhat did the boy drop?\nHis toy.{eos}",
        f"Zoe tied her shoes before running.\nWhat did Zoe tie?\nHer shoes.{eos}",
        f"Liam fed the ducks at the pond.\nWhat did Liam feed?\nThe ducks.{eos}",
        f"The ice melted in the sun.\nWhat melted in the sun?\nThe ice.{eos}",
        f"The leaves fell from the tree.\nWhat fell from the tree?\nThe leaves.{eos}",
        f"The wind blew through the trees.\nWhat blew through the trees?\nThe wind.{eos}",
        f"The boy whispered to his friend.\nWhat did the boy whisper?\n(To his friend.){eos}",  # Optional rephrase
        f"The children laughed at the clown.\nWhat did the children laugh at?\nThe clown.{eos}",
        f"The girl smiled at the camera.\nWhat did the girl smile at?\nThe camera.{eos}",
        f"Tom rode his bike to school.\nWhat did Tom ride?\nHis bike.{eos}",
        f"The truck drove down the road.\nWhat drove down the road?\nThe truck.{eos}",
        f"The frog jumped into the pond.\nWhat jumped into the pond?\nThe frog.{eos}",
        f"The clock ticked loudly.\nWhat ticked loudly?\nThe clock.{eos}",
        f"Jack built a sandcastle on the beach.\nWhat did Jack build?\nA sandcastle.{eos}",
        f"The bear walked through the forest.\nWhat walked through the forest?\nThe bear.{eos}",
        f"The plane flew above the clouds.\nWhat flew above the clouds?\nThe plane.{eos}",
        f"The kangaroo jumped across the path.\nWhat jumped across the path?\nThe kangaroo.{eos}",
        f"The baby crawled across the floor.\nWhat crawled across the floor?\nThe baby.{eos}",
        f"The squirrel hid an acorn under the tree.\nWhat did the squirrel hide?\nAn acorn.{eos}",
        f"The girl drank a glass of milk.\nWhat did the girl drink?\nA glass of milk.{eos}",
        f"The baby crawled across the floor.\nWhat did the baby crawl across?\nThe floor.{eos}",
        f"The owl hooted in the night.\nWhat hooted in the night?\nThe owl.{eos}",
        f"The spider spun a web in the corner.\nWhat did the spider spin?\nA web.{eos}",
        f"The cat hid under the table.\nWhat did the cat hide under?\nThe table.{eos}",
        f"The children built a fort with blankets.\nWhat did the children use to build a fort?\nBlankets.{eos}",
        f"The baby smiled at the teddy bear.\nWhat did the baby smile at?\nThe teddy bear.{eos}",
        f"The sun rose behind the mountains.\nWhat rose behind the mountains?\nThe sun.{eos}",
        f"The stars twinkled in the sky.\nWhat twinkled in the sky?\nThe stars.{eos}",
        f"Alex found a shiny coin on the ground.\nWhat did Alex find?\nA shiny coin.{eos}",
        f"James wrote a letter to his friend.\nWhat did James write?\nA letter.{eos}",
        f"Mark ate a slice of pizza.\nWhat did Mark eat?\nA slice of pizza.{eos}",
        f"Emma ate a sweet cupcake.\nWhat did Emma eat?\nA sweet cupcake.{eos}",
        f"Leo colored the picture with crayons.\nWhat did Leo color?\nThe picture.{eos}",
        f"The goat chewed on some grass.\nWhat did the goat chew on?\nSome grass.{eos}",
        f"The boy opened his birthday gift.\nWhat did the boy open?\nHis birthday gift.{eos}",
        f"The dog barked loudly at the door.\nWhat did the dog bark at?\nThe door.{eos}",
        f"The puppy wagged its tail.\nWhat did the puppy wag?\nIts tail.{eos}",
        f"Zoe tied her shoes before running.\nWhat did Zoe tie?\nHer shoes.{eos}",
        f"The candle flickered in the dark room.\nWhat flickered in the dark room?\nThe candle.{eos}",
        f"The baby smiled at the teddy bear.\nWho did the baby smile at?\nThe teddy bear.{eos}",
        f"The boy whispered to his friend.\nWho did the boy whisper to?\nHis friend.{eos}",
        f"The children laughed at the clown.\nWho did the children laugh at?\nThe clown.{eos}",
        f"The girl smiled at the camera.\nWho smiled at the camera?\nThe girl.{eos}",
        f"James wrote a letter to his friend.\nWho did James write a letter to?\nHis friend.{eos}",
        f"Olivia watched a movie with her brother.\nWho did Olivia watch a movie with?\nHer brother.{eos}",
        f"Zoe tied her shoes before running.\nWho tied her shoes?\nZoe.{eos}",
        f"Mia drew a heart on the paper.\nWho drew a heart?\nMia.{eos}",
        f"Sarah painted a picture of the sun.\nWho painted a picture?\nSarah.{eos}",
        f"Ben threw a stone into the lake.\nWho threw a stone?\nBen.{eos}",
        f"Lily read a storybook before bed.\nWho read a storybook?\nLily.{eos}",
        f"Tom rode his bike to school.\nWho rode a bike to school?\nTom.{eos}",
        f"Max climbed the tall tree.\nWho climbed the tall tree?\nMax.{eos}",
        f"Emma ate a sweet cupcake.\nWho ate a sweet cupcake?\nEmma.{eos}",
        f"Jenny painted her nails pink.\nWho painted her nails?\nJenny.{eos}",
        f"Jack built a sandcastle on the beach.\nWho built a sandcastle?\nJack.{eos}",
        f"David planted flowers in the backyard.\nWho planted flowers?\nDavid.{eos}",
        f"Alex found a shiny coin on the ground.\nWho found a shiny coin?\nAlex.{eos}",
        f"The boy opened his birthday gift.\nWho opened his birthday gift?\nThe boy.{eos}",
        f"Johnny kicked the soccer ball.\nWho kicked the soccer ball?\nJohnny.{eos}",
        f"The dog chased the red ball.\nWho chased the red ball?\nThe dog.{eos}",
        f"The cat chased the ball of yarn.\nWho chased the ball of yarn?\nThe cat.{eos}",
        f"The dog barked loudly at the door.\nWho barked at the door?\nThe dog.{eos}",
        f"Lucy picked a red apple from the tree.\nWho picked a red apple?\nLucy.{eos}",
        f"Anna jumped in the puddle.\nWho jumped in the puddle?\nAnna.{eos}",
        f"Mark ate a slice of pizza.\nWho ate a slice of pizza?\nMark.{eos}",
        f"Sammy found a shell at the beach.\nWho found a shell?\nSammy.{eos}",
        f"Sophia listened to music in her room.\nWho listened to music?\nSophia.{eos}",
        f"James wrote a letter to his friend.\nWho wrote a letter?\nJames.{eos}",
        f"The girl drank a glass of milk.\nWho drank a glass of milk?\nThe girl.{eos}",
        f"Tommy wore a blue hat.\nWho wore a blue hat?\nTommy.{eos}",
        f"Leo colored the picture with crayons.\nWho colored the picture?\nLeo.{eos}",
        f"Sarah made a big sandwich.\nWho made a big sandwich?\nSarah.{eos}",
        f"Liam fed the ducks at the pond.\nWho fed the ducks?\nLiam.{eos}",
        f"Noah threw the ball over the fence.\nWho threw the ball?\nNoah.{eos}",
        f"The kids played tag in the park.\nWho played tag in the park?\nThe kids.{eos}",
        f"The children built a fort with blankets.\nWho built a fort?\nThe children.{eos}",
        f"Olivia watched a movie with her brother.\nWho watched a movie?\nOlivia.{eos}",
        f"The boy rode his skateboard down the hill.\nWho rode his skateboard?\nThe boy.{eos}",
        f"The baby crawled across the floor.\nWho crawled across the floor?\nThe baby.{eos}",
        f"The squirrel hid an acorn under the tree.\nWho hid an acorn?\nThe squirrel.{eos}",
        f"The horse ran across the field.\nWho ran across the field?\nThe horse.{eos}",        
    ]

    return {"text_to_tokenize": texts} 

    
def load_data_with_caching(config, tokenizer):
    cache_path = os.path.join(config.cache_dir, "tokenized_data")
    
    cache_dir = Path(config.cache_dir)
    raw_cache_path = cache_dir / "raw_data"
    tokenized_cache_path = cache_dir / "tokenized_data"
    train_output_path = os.path.join(config.cache_dir, "train_data.txt")

    if tokenized_cache_path.exists():
        print(f"Loading tokenized dataset from {tokenized_cache_path}")
        dataset = load_from_disk(str(tokenized_cache_path))
        if "train" in dataset and "validation" in dataset:
            return dataset["train"], dataset["validation"]
        elif "train" in dataset:
            print("Validation split not found in cached dataset, returning only train.")
            return dataset["train"], None # Or handle this case as needed
        else:
            raise KeyError(f"Cached dataset does not contain 'train' split. Found: {dataset.keys()}")
        
    print("Loading raw dataset...")  

    if config.dataset_name in ["Trivia_qa"]:
        raw_dataset = load_dataset("trivia_qa","unfiltered")
        print("Preparing TriviaQA dataset...")
        dataset = raw_dataset.map(
            lambda x: prepare_trivia_qa(x, tokenizer),
            batched=True,
            num_proc=os.cpu_count(),
        )
    elif config.dataset_name in ["Fairytale"]: # Example other datasets
        raw_dataset = load_dataset("WorkInTheDark/FairytaleQA")
        print("Preparing FairytaleQA dataset...")
        dataset = raw_dataset.map(
            lambda x: default_qa(x, tokenizer),
            batched=True,   
            num_proc=os.cpu_count(),             
        )

    elif config.dataset_name in ["Default"]:
        # Create the dataset from the provided `texts` list
        texts = [
            f"The rabbit hopped through the green field. Where did the rabbit hop? Through the green field.{tokenizer.eos_token}",
            f"The cat slept on the warm couch. Where did the cat sleep? On the warm couch.{tokenizer.eos_token}",
        ]
        data_dict = {"text_to_tokenize": texts}
        raw_dataset = Dataset.from_dict(data_dict)

        print("Preparing Default dataset...")
        dataset = raw_dataset.train_test_split(test_size=0.1)  # dataset is a DatasetDict
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]  # Directly access the "test" split
        dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    else:
        # You should define how to handle other datasets
        raise NotImplementedError(f"Dataset {config.dataset_name} not supported.")

    try:
        # Save the original text of the training split BEFORE tokenization
        with open(train_output_path, "w", encoding="utf-8") as f:
            for item in dataset["train"]:
                f.write(item["text_to_tokenize"] + "\n")
        print(f"Original training data saved to: {train_output_path}")
    except Exception as e:
        pass

    print("Tokenizing dataset...")
    block_size = config.max_seq_length
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, block_size),
        batched=True,
        num_proc=os.cpu_count(),  # Use available CPU cores
    )

    if "train" in dataset and "validation" in dataset:        
        return dataset["train"], dataset["validation"]
    elif "train" in dataset:        
        print(f"Training data saved to: {train_output_path}")
        print("Validation split not created, returning only train.")
        return dataset["train"], None
    else:
        raise ValueError(f"Processed dataset does not contain 'train' split: {dataset.keys()}") 

def load_and_prepare_data(config: TrainingConfig, tokenizer):
    """
    Loads, prepares (using dataset-specific logic), and tokenizes the dataset
    using a standardized two-step approach.
    Returns tokenized train and validation datasets.
    """

    print(f"Starting data loading and preparation...")
    print(f"Loading dataset: {config.dataset_name} ({config.dataset_config_name})")
    try:
        # Load raw dataset (consider adding cache_dir=...)
        # Set trust_remote_code=True if required by the specific dataset
        raw_dataset = load_dataset(config.dataset_name, config.dataset_config_name, trust_remote_code=True)
        print(f"Dataset loaded successfully. Original structure: {raw_dataset}")
    except Exception as e:
        print(f"Fatal: Error loading dataset '{config.dataset_name}'. Exception: {e}")
        raise # Stop execution if dataset loading fails

    # --- 1. Dataset-Specific Preparation ---
    print("Applying dataset-specific preparation...")
    prepared_dataset = DatasetDict()

    # Ensure 'train' split exists before proceeding
    if "train" not in raw_dataset:
        raise ValueError(f"Critical: 'train' split not found in loaded dataset: {raw_dataset}")
    # Get original column names to remove them after preparation
    original_columns = list(raw_dataset['train'].features.keys())
    print(f"Original columns identified: {original_columns}")

    # --- Select the appropriate preparation function ---
    if config.dataset_name == "trivia_qa":
        prepare_fn = prepare_trivia_qa
        print("Selected 'prepare_trivia_qa' function.")
    # Add elif conditions for other datasets you expect to use
    elif config.dataset_name in ["wikitext", "other_dataset_with_text_col"]: # Example other datasets
        prepare_fn = prepare_simple_text_dataset
        print("Selected 'prepare_simple_text_dataset' function.")
    else:
        # Fallback or error handling for unknown datasets
        print(f"Warning: No specific preparation function defined for '{config.dataset_name}'. "
              "Attempting fallback using 'prepare_simple_text_dataset' (assumes a 'text' column).")
        prepare_fn = prepare_simple_text_dataset

    available_splits = list(raw_dataset.keys())
    print(f"Processing available splits: {available_splits}")

    # Apply preparation map to all available splits
    for split_name in available_splits:
        current_split_data = raw_dataset.get(split_name) # Use .get() for safety
        if current_split_data is not None and len(current_split_data) > 0:
            print(f"Preparing split: {split_name}...")
            # Use lambda to pass the tokenizer to the preparation function
            prep_fn_with_tokenizer = lambda exs: prepare_fn(exs, tokenizer)
            try:
                prepared_dataset[split_name] = current_split_data.map(
                    prep_fn_with_tokenizer,
                    batched=True,
                    remove_columns=original_columns, # Remove original columns here
                    num_proc=os.cpu_count(), # Use available CPU cores
                    desc=f"Preparing {split_name}"
                )
                # Verify the expected column was created
                if "text_to_tokenize" not in prepared_dataset[split_name].column_names:
                    print(f"Warning: 'text_to_tokenize' column was NOT created after preparing split '{split_name}'. "
                          "Check the selected prepare_fn logic.")
                else:
                    print(f"Finished preparing '{split_name}'. Columns: {prepared_dataset[split_name].column_names}")

            except Exception as e:
                print(f"Error during preparation map for split '{split_name}': {e}")
                # Decide how to handle: raise error, skip split, etc.
                prepared_dataset[split_name] = None # Mark as failed/empty
                # raise # Option to stop execution on error
        else:
            print(f"Skipping preparation for empty or None split: {split_name}")
            prepared_dataset[split_name] = current_split_data # Keep it as None or empty


    print(f"Dataset structure after preparation step: {prepared_dataset}")

    # --- 2. Determine Final Train, Validation, and Test Splits ---
    # Use the prepared dataset from now on
    if "train" not in prepared_dataset or prepared_dataset["train"] is None:
        raise ValueError("Critical: 'train' split is missing or empty after preparation step.")

    train_data = prepared_dataset.get("train")
    val_data = prepared_dataset.get("validation")
    test_data = prepared_dataset.get("test") # For info saving

    # Create validation split if it doesn't exist or is empty after preparation
    if not val_data and train_data:
        print(f"Validation split missing or empty. Creating new one from train ({config.validation_split_percentage * 100:.1f}%)...")
        if len(train_data) > 1: # Need at least 2 samples to split
            split = train_data.train_test_split(
                test_size=config.validation_split_percentage,
                seed=config.seed,
                shuffle=True # Shuffle before splitting is usually good
            )
            train_data = split["train"]
            val_data = split["test"]
            prepared_dataset["train"] = train_data # Update dict with the new splits
            prepared_dataset["validation"] = val_data
            print("Validation split created.")
        else:
            print("Warning: Train split has <= 1 sample after preparation, cannot create validation split.")
            # Keep val_data as None or empty

    # --- 3. (Optional) Save Dataset Information ---
    # This section accesses the data *before* the final tokenization step
    num_train_samples = len(train_data) if train_data else 0
    num_val_samples = len(val_data) if val_data else 0
    num_test_samples = len(test_data) if test_data else 0

    print(f"Sample counts before tokenization: Train={num_train_samples}, Validation={num_val_samples}, Test={num_test_samples if test_data else 'N/A'}")

    if config.save_path: # Proceed only if a save path is configured
        os.makedirs(config.save_path, exist_ok=True)
        # Sanitize name for filename
        sanitized_dataset_name = re.sub(r'[\\/*?:"<>|]', "", config.dataset_name).replace("/", "_")
        info_filename = f"{sanitized_dataset_name}_info.txt"
        dataset_info_path = os.path.join(config.save_path, info_filename)

        print(f"Attempting to save dataset information to: {dataset_info_path}")
        try:
            with open(dataset_info_path, 'w', encoding='utf-8') as f:
                f.write(f"Dataset Name: {config.dataset_name}\n")
                f.write(f"Dataset Config: {config.dataset_config_name}\n")
                f.write(f"Original Splits Found: {available_splits}\n")
                f.write("--- Sample Counts (After Preparation / Before Tokenization) ---\n")
                f.write(f"Train samples: {num_train_samples}\n")
                f.write(f"Validation samples: {num_val_samples}\n")
                f.write(f"Test samples: {num_test_samples if test_data else 'Not available'}\n")

                # Save prepared text examples (from 'text_to_tokenize')
                f.write("\n--- Sample Prepared Training Examples (First 5) ---\n")
                # Check if train_data exists and has the expected column
                if train_data and num_train_samples > 0 and "text_to_tokenize" in train_data.column_names:
                    # Limit number of examples to save
                    num_examples_to_save = min(5, num_train_samples)
                    for i in range(num_examples_to_save):
                        example_text = train_data[i]["text_to_tokenize"]
                        # Clean for display (replace newlines, strip)
                        cleaned_text = str(example_text).replace('\n', ' \\n ').strip()
                        f.write(f"[{i+1}] {cleaned_text}\n")
                elif train_data and num_train_samples > 0:
                    f.write("Could not find 'text_to_tokenize' column in prepared data to display samples.\n")
                else:
                    f.write("No training data available to preview.\n")
            print(f"Dataset information saved successfully.")
        except Exception as e:
            print(f"Warning: Could not write dataset info file to '{dataset_info_path}'. Error: {e}")
    else:
        print("Skipping dataset info saving as config.save_path is not set.")


    # --- 4. Generic Tokenization ---
    print("Applying generic tokenization using 'generic_tokenize_fn'...")
    tokenized_train_data = None
    tokenized_val_data = None

    # Use a partial function or lambda to pass tokenizer and config to the generic function
    tokenizer_partial_fn = lambda exs: generic_tokenize_fn(exs, tokenizer, config)

    # Process TRAIN data
    if train_data and num_train_samples > 0:
        if "text_to_tokenize" not in train_data.column_names:
            print("Error: Cannot tokenize train data, 'text_to_tokenize' column missing after preparation.")
            # Handle error: maybe return None, None or raise Exception
            tokenized_train_data = None
        else:
            print(f"Tokenizing train data ({num_train_samples} samples)...")
            try:
                tokenized_train_data = train_data.map(
                    tokenizer_partial_fn,
                    batched=True,
                    remove_columns=["text_to_tokenize"], # Remove the intermediate text column
                    num_proc=os.cpu_count(),
                    load_from_cache_file=True, # Use cache if available
                    desc="Tokenizing train data"
                )
                print(f"Finished tokenizing train data. Final columns: {tokenized_train_data.column_names}")
            except Exception as e:
                print(f"Error during tokenization map for train data: {e}")
                tokenized_train_data = None # Mark as failed
                # raise # Option to stop
    else:
        print("Skipping tokenization for empty train data.")

    # Process VALIDATION data
    if val_data and num_val_samples > 0:
        if "text_to_tokenize" not in val_data.column_names:
            print("Error: Cannot tokenize validation data, 'text_to_tokenize' column missing after preparation.")
            tokenized_val_data = None
        else:
            print(f"Tokenizing validation data ({num_val_samples} samples)...")
            try:
                tokenized_val_data = val_data.map(
                    tokenizer_partial_fn,
                    batched=True,
                    remove_columns=["text_to_tokenize"], # Remove intermediate text column
                    num_proc=os.cpu_count(),
                    load_from_cache_file=True, # Use cache if available
                    desc="Tokenizing validation data"
                )
                print(f"Finished tokenizing validation data. Final columns: {tokenized_val_data.column_names}")
            except Exception as e:
                print(f"Error during tokenization map for validation data: {e}")
                tokenized_val_data = None # Mark as failed
                # raise # Option to stop
    else:
        print("Skipping tokenization for empty validation data.")

    # Final status check before returning
    print("Data loading and preparation finished.")
    if tokenized_train_data is None or tokenized_val_data is None:
        print("Warning: Resulting tokenized train or validation data is None due to errors or empty inputs.")
    elif len(tokenized_train_data) == 0 or len(tokenized_val_data) == 0:
        print("Warning: Resulting tokenized train or validation data has zero length.")

    # Return the final tokenized datasets ready for the DataLoader
    return tokenized_train_data, tokenized_val_data

def generic_tokenize_fn(examples, tokenizer, config):
    """
    Tokenizes text from the 'text_to_tokenize' field.
    Assumes 'text_to_tokenize' was created by a preparation step.
    Also creates 'labels' for Causal LM training.
    """
    # Check if the expected column exists and is not empty
    if "text_to_tokenize" not in examples or not examples["text_to_tokenize"]:
        print("Warning: 'text_to_tokenize' column missing or empty in batch for tokenization. Skipping.")
        # Return empty structure matching tokenizer output + labels
        return {'input_ids': [], 'attention_mask': [], 'labels': []}

    # Filter out None or unexpected types just in case preparation failed
    texts = [str(t) for t in examples["text_to_tokenize"] if t is not None]
    if not texts: # If texts list is empty after filtering
        print("Warning: No valid text found in 'text_to_tokenize' after filtering None. Skipping batch.")
        return {'input_ids': [], 'attention_mask': [], 'labels': []}

    # Tokenize the prepared text   
    tokenized_output = tokenizer(
        texts,
        truncation=True,
        padding='max_length',  # Add padding to max_length
        max_length=config.max_seq_length,        
        return_tensors="pt"
    )

    input_ids = tokenized_output["input_ids"]
    labels = input_ids.clone()

    # Shift labels to the right (as the model predicts the next token)
    labels = torch.roll(labels, shifts=-1, dims=1)

    # Set the target for the last token to the ignore index
    labels[:, -1] = -100  # Set to -100 to ignore the last token in loss calculation

    tokenized_output["labels"] = labels
    return tokenized_output


def create_dataloaders(config: TrainingConfig, tokenizer, train_data, val_data):
    """Creates DataLoaders and the DataCollator."""
    # Check if input data is valid before proceeding
    if train_data is None or val_data is None:
        raise ValueError("Cannot create DataLoaders with None train_data or val_data.")
    if len(train_data) == 0 or len(val_data) == 0:
        print("Warning: Creating DataLoaders with empty train or validation dataset.")


    print("Initializing Data Collator...")
    # mlm=False indicates Causal LM. Collator handles padding and labels.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"  # make sure this is specified!
    )

    print("Creating DataLoaders...")
    # Use max() to prevent num_workers > 0 for empty datasets if needed,
    # though DataLoader might handle it. Safest to use 0 if dataset empty.
    train_num_workers = config.num_workers if len(train_data) > 0 else 0
    val_num_workers = config.num_workers if len(val_data) > 0 else 0

    shuffle = True if len(train_data) > 0 else False
    
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        shuffle=shuffle, # No need to shuffle empty data
        num_workers=train_num_workers,
        pin_memory=True # Can speed up CPU to GPU transfers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size, # Can often use larger batch size for validation
        collate_fn=data_collator,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True
    )

    for batch in train_loader:
        print("--- Dummy Loader Batch ---")
        print("Input IDs:", batch["input_ids"])
        print("Labels:", batch["labels"])
        break

    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")

    return train_loader, val_loader, data_collator


