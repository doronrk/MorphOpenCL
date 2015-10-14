age:
    Mode 3 is the 'final' version of the project

    Change the mode with the following keys
        1   time domain. Tip: try producing vocal white noise, then singing a vowel
        2   frequency domain
        3*  face FFT visualizer. Tip: try looking at the face from the side. 
        4   face melt (bug that looked cool)
        5   split (bug that looked cool)

    Notes on Mode 3
        Responds best to white noise and music.
        Use up/down arrow keys to control face ratchetness. Face ratchnetness looks coolest in the range 0.9 to 1.0

    Press 'i' to show/hide the instructions when running the visualizer.

Idea behind Mode 3
    - Construct a 'brightness' matrix with values that reflect the brightness at a given pixel relative to the entire image
    - Send a proportional number of 1,000,000 particles to that point in x,y space with additive brightness to reconstruct the image
    - Project the x,y points onto a cyclinder to give the illusion of a 3-d face
    - Perturb the z position of the particles at given rows along the face according to energy at corresponding frequency bins.

ideas for improvements:
    - normalize height of the spectrum according to most recent volume. 
    - add point lighting of the face
    - better clarity of mapping between precise frequency y-location of face particle perturbance. 
