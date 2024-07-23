import onnxruntime as ort
import numpy as np
import torch
import re


def tokenlize(content):
    content = re.sub(r"([.!?])", r" \1", content)
    content = re.sub(r"[^a-zA-Z.!?]+", r" ", content)
    token = [i.strip().lower() for i in content.split()]
    return token


class Transform():
    def __init__(self):
        self.PAD_TAG = "PAD"
        self.UNK = 0

    def trans(self, sentence, ws, max_len=None):
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        return [ws.get(word, self.UNK) for word in sentence]


temp_transform = Transform()


text_input = "Witchery might just be the most incoherent and lamentably scripted horror movie of the 80's but, luckily enough, \it has a few compensating qualities like fantastic gore effects, an exhilarating musical score and some terrific casting choices. Honestly the screenplay doesn't make one iota of sense, but who cares when Linda Blair (with an exploded hairstyle) portrays yet another girl possessed by evil powers and David Hasselhof depicts a hunky photographer (who can't seem to get laid) in a movie that constantly features bloody voodoo, sewn-shut lips, upside down crucifixions, vicious burnings and an overused but genuinely creepy tune. Eight random people are gathered together on an abandoned vacation resort island off the coast of Massachusetts. The young couple is there to investigate the place's dark history; the dysfunctional family (with a pregnant Linda Blair even though nobody seems to bother about who the father is and what his whereabouts are) considers re-opening the hotel and the yummy female architect simply tagged along for casual sex. They're forced to stay the night in the ramshackle hotel and then suddenly the previous landlady an aging actress or something who always dresses in black starts taking them out in various engrossing ways. Everything is somehow related to the intro sequence showing a woman accused of witchery jump out of a window. Anyway, the plot is definitely of minor importance in an Italian horror franchise that started as an unofficial spin-off of The Evil Dead. The atmosphere is occasionally unsettling and the make-up effects are undoubtedly the most superior element of the entire film. There's something supremely morbid and unsettling about staring at a defenseless woman hanging upside down a chimney and waiting to get fried."



labels = ['negative', 'positive']

index_to_label = {idx: label for idx, label in enumerate(labels)}
# np.save('../data/index_to_label.npy', index_to_label )

# index_to_label = np.load("../data/index_to_label.npy", allow_pickle=True).item()
ws = np.load("../data/aclImdb_trans.npy", allow_pickle=True).item()

tokens = tokenlize(text_input)
indices = temp_transform.trans(sentence=tokens, ws=ws, max_len=200 )

input_tensor = torch.tensor(indices, dtype=torch.long).view(1, -1).numpy()

onnx_path = "../data/new_LSTM_model.onnx"
session = ort.InferenceSession(onnx_path)

input_name = session.get_inputs()[0].name
inputs = {input_name: input_tensor}

output = session.run(None, inputs)


output_name = session.get_outputs()[0].name

output_tensor = output[0]
predicted_class_index = np.argmax(output_tensor, axis=1)[0]

predicted_label = index_to_label.get(predicted_class_index, "Unknown")

print(f" {predicted_label}")


