import torch

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
    
def create_default_data(tokenizer, config, split="train", max_samples=100): 


    all_input_ids = []
    all_attention_masks = []
    all_targets = []

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
