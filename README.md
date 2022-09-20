# words2drums
Companion code to the preprint "Setting the rhythm scene: deep learning-based drum loop generation from arbitrary language cues"

First ensure you have the required libraries installed:

    keras

Unpack the pre-trained model:

    $ tar xvfz trained_words2drums.tar.gz

Add any word or phrase to `inputs.txt`, the script will generate one drum pattern per line:

    $ python3 words2drums.py

If you have any questions, please contact the author at ignacio.tripodi (at) colorado.edu.

Citations:

*Tripodi, I.J., "Setting the rhythm scene: deep learning-based drum loop generation from arbitrary language cues" 
