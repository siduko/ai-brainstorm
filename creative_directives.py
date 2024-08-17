"""

This file contains the creative directives that will be used to generate ideas.

Each is a strategy for generating ideas.

Each is comprised of a name, the instruction the LLM should follow, and a
description of what is meant by the strategy.

```
"Directive Name": (
    "Instruction",
    "Description",
)
```

"""

CREATIVE_DIRECTIVES = {
    "Conceptual Blend": (
        "Combine two seemingly unrelated concepts or domains.",
        "Explore unexpected synergies between disparate ideas.",
    ),
    "Perspective Shift": (
        "Approach the problem from an entirely different point of view.",
        "Reimagine the concept through an unexpected lens.",
    ),
    "Amplify Extremes": (
        "Take a key aspect of the idea and push it to its logical extreme.",
        "Explore the boundaries of the concept by maximizing certain elements.",
    ),
    "Invert Assumptions": (
        "Challenge and reverse the core assumptions of the current approach.",
        "Flip key aspects of the idea on their head.",
    ),
    "Temporal Dynamics": (
        "Consider how the idea might evolve or adapt over time.",
        "Explore the concept's past, present, and future iterations.",
    ),
    "Synaptic Leap": (
        "Make an unexpected connection between disparate elements.",
        "Bridge seemingly unrelated aspects of the problem or solution.",
    ),
    "Fractal Thinking": (
        "Apply the core concept at different scales simultaneously.",
        "Consider how the idea manifests at micro and macro levels.",
    ),
    "Empathic Reimagining": (
        "Redesign the idea with deep empathy for a specific user or stakeholder.",
        "Center the concept around the needs and experiences of others.",
    ),
    "Sensory Transmutation": (
        "Translate the concept into a different sensory modality.",
        "Reimagine the idea through alternative sensory experiences.",
    ),
    "Quantum Superposition": (
        "Explore contradictory aspects of the idea simultaneously.",
        "Embrace paradoxical elements within the concept.",
    ),
    "Ecosystem Integration": (
        "Consider how the idea fits into and influences a larger system.",
        "Explore the ripple effects and interconnections of the concept.",
    ),
    "Narrative Reframing": (
        "Cast the concept as part of a compelling story or journey.",
        "Embed the idea within a larger narrative or transformative arc.",
    ),
    "Adaptive Resilience": (
        "Explore how the idea could adapt to unexpected challenges.",
        "Consider the concept's flexibility and responsiveness to change.",
    ),
    "Cross-Pollination": (
        "Infuse elements from a completely different field or discipline.",
        "Introduce concepts from unrelated domains to spark new insights.",
    ),
}
