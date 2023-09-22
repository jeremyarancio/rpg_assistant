import logging
from typing import Mapping, Optional

import streamlit as st
from transformers import AutoTokenizer

from fireball_llm_inference import FireballInference 

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

HISTORY = """The creature, a living crucible of coldest winter, rears back; there is a hitching in its neck, and then it vomits out a spray of biting ice...

There is a low hiss; it seems to realise its ice will be no use. Claws like scythes sink into the frozen earth below and flex in readiness...

Said only movement of Lis' turn, will assume applies)

moving forward, only allowing range for herself, her halberd swings out with practiced, deadly precision.
"""

HYPERPARAMATERS = {
        "max_new_tokens": 50,
        "do_sample": True,
        "temperature": 0.2,
        "early_stopping": True,
        "repetition_penalty": float(5),
        "no_repeat_ngram_size": 3 
    }


def generate_prediction(
        last_utterance: str,
        command_descr: str,
        history: str,
        hyperparameters: Mapping,
    ) -> str:
    """Generate prediction for next utterance"""
    fireball_inference = FireballInference()
    next_utterance = fireball_inference.generate_next_utterance(
        last_utterance,
        command_descr,
        history,
        hyperparameters
    )
    return next_utterance


if __name__ == "__main__":

    col1, col2 = st.columns(2)
    with col1:
        st.text_input(
            label="Last utterance", 
            value="Razored teeth lash out to take advantage of an opening", 
            key="last_utterance"
        )
    with col2:
        st.text_input(
            label="History", 
            value=HISTORY, 
            key="history"
        )
        st.text_input(
            label="Command descr.", 
            value="Lisbeth Ethuil uses Polearm Master - Bonus Attack!",
            key="command_descr"
        )

    if st.button("Generate"):
        with st.spinner("Generating story..."):
            next_utterance = generate_prediction(
                last_utterance=st.session_state['last_utterance'],
                command_descr=st.session_state['command_descr'],
                history=st.session_state['history'],
                hyperparameters=HYPERPARAMATERS
            )
            st.text(next_utterance)
