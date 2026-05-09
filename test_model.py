from model import HindiGPT

model = HindiGPT()

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Total Parameters: {total_params:,}")
