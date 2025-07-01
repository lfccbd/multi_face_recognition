.. figure:: https://winnerschapelcbd.com/static/assets/img/logo.png
    :align: center


==========================
Winners Int. JHB CBD
==========================

Winners Chapel JHB CBD is part of the Living Faith Church Worldwide, a mandate By God 
Through his Servant Bishop David Oyedepo for the Liberation of mankind. As a Commission, 
we have experienced amazing testimonies ever since this commission was handed down - 
that is over thirty years now, To God be the glory.


==========
Setup
==========

Install package dependencies

.. code-block:: bash

  pip install -r requirements.txt


Additional Setup
-----------------

Ensure the required faces to detect are in the *known_faces* directory. Ensure that image file names matches 
the expected names to be shown during detection. For example if your file is titled "demo.jpg", once the face is 
detected, the name shown will be "demo".


Runing Code
-------------

Run Detection
^^^^^^^^^^^^^^^^

Inital startup will train and save the encoding facenet as a pickle file which can later be used. 

.. code-block:: bash

    python recognition.py


Re-encoding
^^^^^^^^^^^^^^

Update your *known_faces* directory and run code to re-encode and get new pickle file. 
This command also rund the detection after encoding.

.. code-block:: bash

    python recognition.py --re-encode
