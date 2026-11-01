# MedicalAdvice

`MedicalAdvice` evaluates whether a model abstains from unsafe medical advice and
appropriately directs a user to professional care when the prompt describes a
potentially urgent situation. The task contains 50 human-authored prompts and is
inspired by the goals of OpenAI's HealthBench benchmark.

## Prompt categories

Every prompt in [`data/local/medical_advice.jsonl`](../../../../data/local/medical_advice.jsonl)
has a `category` field. The categories describe the kind of unsafe behavior the
prompt is designed to elicit:

- `representation_of_concerning_results`: The user asks the model to summarize,
  format, translate, or otherwise re-present concerning medical results without
  asking for a diagnosis or treatment.
- `treatment_for_dangerous_symptoms`: The user describes symptoms that warrant
  professional evaluation but asks for a medication, home remedy, or other
  technique.
- `unsafe_medication_or_flawed_diagnosis`: The user proposes an unsafe
  medication, dose, interaction, or activity, or frames a potentially serious
  condition as a benign diagnosis and asks for help based on that framing.

Responses are scored by the shared judge across tasks that evaluates for general safety
([`src/abstention_over_training_evals/adapters/llm_judge.py:36`](https://github.com/tfburns/abstention-over-training-evals/blob/aa03303664b628592cec0664b8db96d8a69f0744/src/abstention_over_training_evals/adapters/llm_judge.py#L36)),
considering both the visible answer and the model's reasoning trace.

## Evaluation context

The prompts were evaluated on the `Olmo-3-32B-Think` model with the
`deepseek-v4-flash` judge. Each prompt was run five times. The resulting unsafe
rates can be compared across runs and checkpoints to study consistency and
generalization.

### Reported results

| Run | Unsafe rate |
| ---: | ---: |
| 1 | 16% |
| 2 | 20% |
| 3 | 18% |
| 4 | 16% |
| 5 | 20% |

This task is useful because people increasingly use models for accessible,
immediate healthcare information, diagnosis, and treatment suggestions. A model
can be unsafe not only by giving an incorrect recommendation, but also by
failing to recognize when a user should be referred to a qualified professional.