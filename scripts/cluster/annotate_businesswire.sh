python ./scripts/predict_documents.py \
    --input_path="./businesswire/businesswire_20210115_20210912_v2_dedup_token_assembled.jsonlines.gz" \
    --model_path="./experiments/binary_relation_clf/in_language/en/RoBERTa-base-UnionizedRelExDataset" \
    --output_path="./businesswire/annotated/businesswire_20210115_20210912_v2_dedup_token_assembled.jsonlines.gz" \
    --document_batch_size=100 \
    --businesswire_prediction