final direction:
    - perterb the face based on the spectrum

ideas for improvements:
    - normalize height of the spectrum according to most recent volume
    - currently using a buffer size of 2048
        - would rather use a smaller buffer size so that its more responsive, but in its current state, that causes the spectrogram to look really sparse
        - but before I can do this I need to write code to send some fraction of the particles to interpolated heights between bin
        - it's drawing 1 million points right now, so a ton of them are redundant in the spectrum view

demo:
    - well calibrated for When I'm Small