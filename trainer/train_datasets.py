import torch
from datasets import load_dataset  # Hugging Face datasets library
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BatchEncoding

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, targets):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.targets[idx]
        }
    
class TinyStoriesDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['labels'][idx]
        }
    
    
def default_dataset(tokenizer, config, split="train", max_samples=100):

    all_input_ids = []
    all_attention_masks = []
    all_targets = []

    eos = tokenizer.eos_token if tokenizer.eos_token else ""

    texts = [
        f"The rabbit hopped through the green field. Where did the rabbit hop? Through the green field.{eos}",
        f"The rabbit hopped over the trunk. Where did the rabbit hop? Over the trunk.{eos}",
        f"The cat slept on the warm couch. Where did the cat sleep? On the warm couch.{eos}",
        f"The dog chased the red ball. What did the dog chase? The red ball.{eos}",
        f"The bird sang on the tree. Where did the bird sing? On the tree.{eos}",
        f"Tom rode his bike to school. Where did Tom ride his bike? To school.{eos}",
        f"The fish swam in the pond. Where did the fish swim? In the pond.{eos}",
        f"Emma ate a sweet cupcake. What did Emma eat? A sweet cupcake.{eos}",
        f"The butterfly flew over the flowers. What did the butterfly fly over? The flowers.{eos}",
        f"The teddy bear sat on the bed. Where was the teddy bear? On the bed.{eos}",
        f"Lucy picked a red apple from the tree. What did Lucy pick? A red apple.{eos}",
        f"Johnny kicked the soccer ball. What did Johnny kick? The soccer ball.{eos}",
        f"Max climbed the tall tree. What did Max climb? The tall tree.{eos}",
        f"The dog barked loudly at the door. What did the dog bark at? The door.{eos}",
        f"The boy rode his skateboard down the hill. What did the boy ride? His skateboard.{eos}",
        f"Sarah made a big sandwich. What did Sarah make? A big sandwich.{eos}",
        f"The baby smiled at the teddy bear. Who did the baby smile at? The teddy bear.{eos}",
        f"The boy opened his birthday gift. What did the boy open? His birthday gift.{eos}",
        f"The cat chased the ball of yarn. What did the cat chase? The ball of yarn.{eos}",
        f"Tommy wore a blue hat. What did Tommy wear? A blue hat.{eos}",
        f"The flowers bloomed in the garden. Where did the flowers bloom? In the garden.{eos}",
        f"The moon shone brightly at night. When did the moon shine? At night.{eos}",
        f"The stars twinkled in the sky. Where did the stars twinkle? In the sky.{eos}",
        f"The truck drove down the road. Where did the truck drive? Down the road.{eos}",
        f"A bird flew in the sky. Where did the bird fly? In the sky.{eos}",
        f"The bear walked through the forest. Where did the bear walk? Through the forest.{eos}",
        f"Sarah painted a picture of the sun. What did Sarah paint? A picture of the sun.{eos}",
        f"The dog slept in its bed. Where did the dog sleep? In its bed.{eos}",
        f"The rabbit nibbled on a carrot. What did the rabbit nibble on? A carrot.{eos}",
        f"The horse ran across the field. Where did the horse run? Across the field.{eos}",
        f"Anna jumped in the puddle. What did Anna jump in? The puddle.{eos}",
        f"The clock ticked loudly. What ticked loudly? The clock.{eos}",
        f"Jack built a sandcastle on the beach. Where did Jack build a sandcastle? On the beach.{eos}",
        f"The squirrel hid an acorn under the tree. Where did the squirrel hide an acorn? Under the tree.{eos}",
        f"The girl drank a glass of milk. What did the girl drink? A glass of milk.{eos}",
        f"Ben threw a stone into the lake. What did Ben throw? A stone.{eos}",
        f"The baby crawled across the floor. Where did the baby crawl? Across the floor.{eos}",
        f"Lily read a storybook before bed. What did Lily read? A storybook.{eos}",
        f"The wind blew through the trees. What blew through the trees? The wind.{eos}",
        f"Alex found a shiny coin on the ground. What did Alex find? A shiny coin.{eos}",
        f"The kids played tag in the park. Where did the kids play tag? In the park.{eos}",
        f"Jenny painted her nails pink. What did Jenny paint? Her nails.{eos}",
        f"The frog jumped into the pond. Where did the frog jump? Into the pond.{eos}",
        f"The sun rose behind the mountains. Where did the sun rise? Behind the mountains.{eos}",
        f"David planted flowers in the backyard. Where did David plant flowers? In the backyard.{eos}",
        f"The owl hooted in the night. When did the owl hoot? In the night.{eos}",
        f"Mia drew a heart on the paper. What did Mia draw? A heart.{eos}",
        f"The kitten climbed onto the windowsill. Where did the kitten climb? Onto the windowsill.{eos}",
        f"Noah threw the ball over the fence. Where did Noah throw the ball? Over the fence.{eos}",
        f"Sammy found a shell at the beach. What did Sammy find? A shell.{eos}",
        f"The leaves fell from the tree. What fell from the tree? The leaves.{eos}",
        f"The boy whispered to his friend. Who did the boy whisper to? His friend.{eos}",
        f"Lucy danced in the rain. Where did Lucy dance? In the rain.{eos}",
        f"The candle flickered in the dark room. Where did the candle flicker? In the dark room.{eos}",
        f"Mark ate a slice of pizza. What did Mark eat? A slice of pizza.{eos}",
        f"The children laughed at the clown. Who did the children laugh at? The clown.{eos}",
        f"The rain fell on the roof. Where did the rain fall? On the roof.{eos}",
        f"Olivia watched a movie with her brother. What did Olivia watch? A movie.{eos}",
        f"The goat chewed on some grass. What did the goat chew on? Some grass.{eos}",
        f"James wrote a letter to his friend. What did James write? A letter.{eos}",
        f"The plane flew above the clouds. Where did the plane fly? Above the clouds.{eos}",
        f"The children built a fort with blankets. What did the children build? A fort.{eos}",
        f"Sophia listened to music in her room. What did Sophia listen to? Music.{eos}",
        f"The cat hid under the table. Where did the cat hide? Under the table.{eos}",
        f"The ice melted in the sun. What melted in the sun? The ice.{eos}",
        f"Liam fed the ducks at the pond. What did Liam feed? The ducks.{eos}",
        f"The boy dropped his toy on the floor. What did the boy drop? His toy.{eos}",
        f"Zoe tied her shoes before running. What did Zoe tie? Her shoes.{eos}",
        f"The spider spun a web in the corner. Where did the spider spin a web? In the corner.{eos}",
        f"Emma opened the window. What did Emma open? The window.{eos}",
        f"The kangaroo jumped across the path. Where did the kangaroo jump? Across the path.{eos}",
        f"Leo colored the picture with crayons. What did Leo color? The picture.{eos}",
        f"The girl smiled at the camera. What did the girl smile at? The camera.{eos}",
        f"The puppy wagged its tail. What did the puppy wag? Its tail.{eos}",
        f"The cat chased the ball of yarn. What did the cat chase? The ball of yarn.{eos}",
        f"Tommy wore a blue hat. What did Tommy wear? A blue hat.{eos}",
        f"Sarah painted a picture of the sun. What did Sarah paint? A picture of the sun.{eos}",
        f"David planted flowers in the backyard. What did David plant? Flowers.{eos}",
        f"Mia drew a heart on the paper. What did Mia draw? A heart.{eos}",
        f"The boy dropped his toy on the floor. What did the boy drop? His toy.{eos}",
        f"Zoe tied her shoes before running. What did Zoe tie? Her shoes.{eos}",
        f"Liam fed the ducks at the pond. What did Liam feed? The ducks.{eos}",
        f"The ice melted in the sun. What melted in the sun? The ice.{eos}",
        f"The leaves fell from the tree. What fell from the tree? The leaves.{eos}",
        f"The wind blew through the trees. What blew through the trees? The wind.{eos}",
        f"The boy whispered to his friend. What did the boy whisper? (To his friend.){eos}",  # Optional rephrase
        f"The children laughed at the clown. What did the children laugh at? The clown.{eos}",
        f"The girl smiled at the camera. What did the girl smile at? The camera.{eos}",
        f"Tom rode his bike to school. What did Tom ride? His bike.{eos}",
        f"The truck drove down the road. What drove down the road? The truck.{eos}",
        f"The frog jumped into the pond. What jumped into the pond? The frog.{eos}",
        f"The clock ticked loudly. What ticked loudly? The clock.{eos}",
        f"Jack built a sandcastle on the beach. What did Jack build? A sandcastle.{eos}",
        f"The bear walked through the forest. What walked through the forest? The bear.{eos}",
        f"The plane flew above the clouds. What flew above the clouds? The plane.{eos}",
        f"The kangaroo jumped across the path. What jumped across the path? The kangaroo.{eos}",
        f"The baby crawled across the floor. What crawled across the floor? The baby.{eos}",
        f"The squirrel hid an acorn under the tree. What did the squirrel hide? An acorn.{eos}",
        f"The girl drank a glass of milk. What did the girl drink? A glass of milk.{eos}",
        f"The baby crawled across the floor. What did the baby crawl across? The floor.{eos}",
        f"The owl hooted in the night. What hooted in the night? The owl.{eos}",
        f"The spider spun a web in the corner. What did the spider spin? A web.{eos}",
        f"The cat hid under the table. What did the cat hide under? The table.{eos}",
        f"The children built a fort with blankets. What did the children use to build a fort? Blankets.{eos}",
        f"The baby smiled at the teddy bear. What did the baby smile at? The teddy bear.{eos}",
        f"The sun rose behind the mountains. What rose behind the mountains? The sun.{eos}",
        f"The stars twinkled in the sky. What twinkled in the sky? The stars.{eos}",
        f"Alex found a shiny coin on the ground. What did Alex find? A shiny coin.{eos}",
        f"James wrote a letter to his friend. What did James write? A letter.{eos}",
        f"Mark ate a slice of pizza. What did Mark eat? A slice of pizza.{eos}",
        f"Emma ate a sweet cupcake. What did Emma eat? A sweet cupcake.{eos}",
        f"Leo colored the picture with crayons. What did Leo color? The picture.{eos}",
        f"The goat chewed on some grass. What did the goat chew on? Some grass.{eos}",
        f"The boy opened his birthday gift. What did the boy open? His birthday gift.{eos}",
        f"The dog barked loudly at the door. What did the dog bark at? The door.{eos}",
        f"The puppy wagged its tail. What did the puppy wag? Its tail.{eos}",
        f"Zoe tied her shoes before running. What did Zoe tie? Her shoes.{eos}",
        f"The candle flickered in the dark room. What flickered in the dark room? The candle.{eos}",
        f"The baby smiled at the teddy bear. Who did the baby smile at? The teddy bear.{eos}",
        f"The boy whispered to his friend. Who did the boy whisper to? His friend.{eos}",
        f"The children laughed at the clown. Who did the children laugh at? The clown.{eos}",
        f"The girl smiled at the camera. Who smiled at the camera? The girl.{eos}",
        f"James wrote a letter to his friend. Who did James write a letter to? His friend.{eos}",
        f"Olivia watched a movie with her brother. Who did Olivia watch a movie with? Her brother.{eos}",
        f"Zoe tied her shoes before running. Who tied her shoes? Zoe.{eos}",
        f"Mia drew a heart on the paper. Who drew a heart? Mia.{eos}",
        f"Sarah painted a picture of the sun. Who painted a picture? Sarah.{eos}",
        f"Ben threw a stone into the lake. Who threw a stone? Ben.{eos}",
        f"Lily read a storybook before bed. Who read a storybook? Lily.{eos}",
        f"Tom rode his bike to school. Who rode a bike to school? Tom.{eos}",
        f"Max climbed the tall tree. Who climbed the tall tree? Max.{eos}",
        f"Emma ate a sweet cupcake. Who ate a sweet cupcake? Emma.{eos}",
        f"Jenny painted her nails pink. Who painted her nails? Jenny.{eos}",
        f"Jack built a sandcastle on the beach. Who built a sandcastle? Jack.{eos}",
        f"David planted flowers in the backyard. Who planted flowers? David.{eos}",
        f"Alex found a shiny coin on the ground. Who found a shiny coin? Alex.{eos}",
        f"The boy opened his birthday gift. Who opened his birthday gift? The boy.{eos}",
        f"Johnny kicked the soccer ball. Who kicked the soccer ball? Johnny.{eos}",
        f"The dog chased the red ball. Who chased the red ball? The dog.{eos}",
        f"The cat chased the ball of yarn. Who chased the ball of yarn? The cat.{eos}",
        f"The dog barked loudly at the door. Who barked at the door? The dog.{eos}",
        f"Lucy picked a red apple from the tree. Who picked a red apple? Lucy.{eos}",
        f"Anna jumped in the puddle. Who jumped in the puddle? Anna.{eos}",
        f"Mark ate a slice of pizza. Who ate a slice of pizza? Mark.{eos}",
        f"Sammy found a shell at the beach. Who found a shell? Sammy.{eos}",
        f"Sophia listened to music in her room. Who listened to music? Sophia.{eos}",
        f"James wrote a letter to his friend. Who wrote a letter? James.{eos}",
        f"The girl drank a glass of milk. Who drank a glass of milk? The girl.{eos}",
        f"Tommy wore a blue hat. Who wore a blue hat? Tommy.{eos}",
        f"Leo colored the picture with crayons. Who colored the picture? Leo.{eos}",
        f"Sarah made a big sandwich. Who made a big sandwich? Sarah.{eos}",
        f"Liam fed the ducks at the pond. Who fed the ducks? Liam.{eos}",
        f"Noah threw the ball over the fence. Who threw the ball? Noah.{eos}",
        f"The kids played tag in the park. Who played tag in the park? The kids.{eos}",
        f"The children built a fort with blankets. Who built a fort? The children.{eos}",
        f"Olivia watched a movie with her brother. Who watched a movie? Olivia.{eos}",
        f"The boy rode his skateboard down the hill. Who rode his skateboard? The boy.{eos}",
        f"The baby crawled across the floor. Who crawled across the floor? The baby.{eos}",
        f"The squirrel hid an acorn under the tree. Who hid an acorn? The squirrel.{eos}",
        f"The horse ran across the field. Who ran across the field? The horse.{eos}",        
    ]

    for text in texts:
        tokenized_text = tokenizer(
            text,
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=config.max_seq_length, 
            add_special_tokens=True
        )

        input_ids = tokenized_text['input_ids'].squeeze(0)      # Shape [T]
        attention_mask = tokenized_text['attention_mask'].squeeze(0) # Shape [T]

        # --- MODIFICATION START ---

        # Create targets with the same shape as input_ids [T]
        targets = input_ids.clone()

        targets[:-1] = input_ids[1:]
        targets[-1] = tokenizer.pad_token_id  # Last token doesn't predict anything

        # Ignore pad tokens in the loss
        targets[targets == tokenizer.pad_token_id] = -100

        # --- MODIFICATION END ---

        all_input_ids.append(input_ids)           # Appending shape [T]
        all_attention_masks.append(attention_mask) # Appending shape [T]
        all_targets.append(targets)             # Appending shape [T]

    # Stack everything to make a proper dataset
    all_input_ids = torch.stack(all_input_ids)
    all_attention_masks = torch.stack(all_attention_masks)
    all_targets = torch.stack(all_targets)

    return MyDataset(all_input_ids, all_attention_masks, all_targets)


def load_tinystories_dataset(tokenizer, config, split="train", max_samples=None, show_progress=True):
    # Load dataset with streaming if it's large
    streaming = max_samples is None or max_samples > 10000
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=streaming)
    
    # Apply sample limit if specified
    if max_samples is not None:
        if streaming:
            dataset = dataset.take(max_samples)
        else:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Initialize progress bar
    if show_progress:
        total = max_samples if max_samples is not None else len(dataset) if not streaming else None
        pbar = tqdm(total=total, desc="Processing Tiny Stories")
    
    # Prepare texts with BOS and EOS tokens
    bos_token = tokenizer.bos_token or ""
    eos_token = tokenizer.eos_token or ""
    texts = []
    
    for example in dataset:
        # Add BOS at beginning and EOS at end
        texts.append(f"{bos_token}{example['text']}{eos_token}")
        if show_progress:
            pbar.update(1)
    
    if show_progress:
        pbar.close()
    
    # Batch tokenization
    if show_progress:
        print("Tokenizing texts...")
    
    tokenized = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=config.max_seq_length,
        return_tensors='pt',
        add_special_tokens=False  # We manually added special tokens
    )
    
    # Create labels for language modeling
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    
    # Shift input_ids to create labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = tokenizer.pad_token_id
    
    # Set padding tokens to -100 to ignore them in loss
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Create the dataset
    encodings = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    
    return TinyStoriesDataset(encodings)

    
