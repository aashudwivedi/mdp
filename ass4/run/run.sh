#!/usr/bin/env bash
javac -cp ./:RLDM_HW4_Tester.jar RunJson.java
python mdp_json.py > mymdp.json
java -cp ./:RLDM_HW4_Tester.jar RunJson mymdp.json
