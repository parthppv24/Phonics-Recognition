# Phonics Recognition API

This is custom API which is created to check phonic pronunciations

It compare preprocesses audio, converts into tensor embeddings using WaveLM model and compares the input audio to reference embeddings  and based on set thresholds calculates the average similarity to all those thresholds to check pronunciation is correct or not

Reference Embeddings for each letter are different and each letter is compared to its own embeddings 

>[!CAUTION]
>This program is based on similarity not multi class classification 
>This does not give confidance score type metric of which letter is being pronunced it just compares to the letters own embeddings and gives similarity based on the custom set threshold which are not dynamic and were found out using a custom audio dataset created 



