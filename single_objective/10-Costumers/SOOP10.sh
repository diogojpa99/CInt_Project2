#!/usr/bin/env bash

(python WHCentral_Ord50.py &
python WHCentral_OrdFile.py &
python WHCorner_Ord50.py &
python WHCorner_OrdFile.py &) && python plots.py
