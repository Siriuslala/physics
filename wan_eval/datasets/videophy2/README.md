---
license: mit
task_categories:
- video-classification
tags:
- multimodal
- video
- text_to_video
- physicalAI
pretty_name: videophy2
size_categories:
- 1K<n<10K
---

Project: https://github.com/Hritikbansal/videophy/tree/main/VIDEOPHY2

```
caption: original prompt in the dataset
video_url: generated video (using original prompt or upsampled caption, depending on the video model)
sa: semantic adherence score (1-5) from human evaluation
pc: physical commonsense score (1-5) from human evaluation
joint: computed as sa >= 4, pc >= 4
physics_rules_followed: list of physics rules followed in the video as judged by human annotators (1)
physics_rules_unfollowed: list of physics rules violated in the video as judged by human annotators (0)
physics_rules_cannot_be_determined: list of physics rules that cannot be grounded in the video by human annotators (2)
human_violated_rules: extra physical rules that are violated in the video as written by human annotators (0)
action: type of action
category: physical activity or object interactions
is_hard: does the instance belong to the hard subset
metadata_rules: mapping between physical rules and their physical law (created by Gemini-Flash)
upsampled_caption: upsampled version of the original prompt (it is used to generate videos for the models that support dense captions)
```

We have uploaded the prompts and upsampled prompts separately at: https://huggingface.co/datasets/videophysics/videophy2_upsampled_prompts