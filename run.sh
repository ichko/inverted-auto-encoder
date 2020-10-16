for msg_size in 2048
do
    echo Running for msg_size=${msg_size}
    python3.7 -m pipelines.glyph_ae ${msg_size}
    echo Done
done

echo Done with all!!!
