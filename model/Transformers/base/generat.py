import torch
import torch.nn.functional as F
from ...body import BodyModel

class TextGenerator(BodyModel):
    def generate_text(self, first_token_idxs, text_size=100, temperature=1.0, top_k=0, top_p=0.0):
        self.eval()
        text = first_token_idxs.copy()
        first_tensors = torch.tensor(text, device=self.device).view(1, -1)

        with torch.no_grad():
            while len(text) < text_size:
                input_tensors = first_tensors[:, -self.ws:] if len(text) > self.ws else first_tensors

                pred = self.predict(input_tensors, return_hidden=False)
                logits = pred[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                    pred_word = top_k_indices[0, torch.multinomial(top_k_probs, num_samples=1)].item()

                elif top_p > 0.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = sorted_probs.cumsum(dim=-1)
                    top_p_mask = cumulative_probs <= top_p
                    top_p_probs = sorted_probs * top_p_mask.float()
                    top_p_probs = top_p_probs / top_p_probs.sum(dim=-1, keepdim=True)
                    pred_word = sorted_indices[0, torch.multinomial(top_p_probs, num_samples=1)].item()

                else:
                    pred_word = torch.multinomial(probs, num_samples=1).item()

                text.append(pred_word)
                first_tensors = torch.cat((first_tensors, torch.tensor([[pred_word]], device=self.device)), dim=1)

        return text
