# for msg_size in 2048
# do
#     echo Running for msg_size=${msg_size}
#     python3.7 -m pipelines.glyph_ae ${msg_size}
#     echo Done
# done

# echo Done with all!!!


for i in 1 2 3 4 5
do
    echo RUN

    python3.7 -m pipelines.iae_classifier True True
    python3.7 -m pipelines.iae_classifier True False
    python3.7 -m pipelines.iae_classifier False True
    python3.7 -m pipelines.iae_classifier False False

    echo Done
done

echo Done with all!!!
