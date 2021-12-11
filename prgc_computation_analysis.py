from ptflops import get_model_complexity_info

from PRGC.model import MultiNonLinearClassifier

# from transformers import BertConfig, BertModel


batch_size = 10
potential_target_relation = 3

drop_prob = 0.3
seq_tag_size = 3
rel_num = 216
embedding_dimention = 768
max_seq_len = 100

# Warning: works only with the model input as tensor with float values

# pretrain model
# bert_config = BertConfig.from_json_file(os.path.join('pretrain_models/bert_base_cased/bert_config.json'))
# bert = BertModel(bert_config)
# macs, params = get_model_complexity_info(bert, ((batch_size, max_seq_len), (batch_size, max_seq_len)), as_strings=False, print_per_layer_stat=True, verbose=True)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# rel_embedding = nn.Embedding(rel_num, embedding_dimention)
# macs, params = get_model_complexity_info(rel_embedding, (batch_size * potential_target_relation, 1), as_strings=False, print_per_layer_stat=True, verbose=True)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# relation judgement
rel_judgement = MultiNonLinearClassifier(embedding_dimention, rel_num, drop_prob)
dimension_rel = (batch_size, embedding_dimention)
rel_judgement_macs, rel_judgement_params = get_model_complexity_info(rel_judgement, dimension_rel, as_strings=False, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', rel_judgement_macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', rel_judgement_params))

# sequence tagging
dimension_seq = (batch_size * potential_target_relation, embedding_dimention * 2)

sequence_tagging_sub = MultiNonLinearClassifier(embedding_dimention * 2, seq_tag_size, drop_prob)
sequence_tagging_sub_macs, sequence_tagging_sub_params = get_model_complexity_info(sequence_tagging_sub, dimension_seq, as_strings=False, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', sequence_tagging_sub_macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', sequence_tagging_sub_params))

sequence_tagging_obj = MultiNonLinearClassifier(embedding_dimention * 2, seq_tag_size, drop_prob)
sequence_tagging_obj_macs, sequence_tagging_obj_params = get_model_complexity_info(sequence_tagging_obj, dimension_seq, as_strings=False, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', sequence_tagging_obj_macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', sequence_tagging_obj_params))

# Skipping as concat fusion is used
# sequence_tagging_sum = SequenceLabelForSO(embedding_dimention, seq_tag_size, drop_prob)

# global correspondence
global_corres = MultiNonLinearClassifier(embedding_dimention * 2, 1, drop_prob)
dimension_corres = (batch_size * max_seq_len * max_seq_len, embedding_dimention * 2)
global_corres_macs, global_corres_params = get_model_complexity_info(global_corres, dimension_corres, as_strings=False, print_per_layer_stat=True,
                                                                     verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', global_corres_macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', global_corres_params))

total_macs = rel_judgement_macs + sequence_tagging_sub_macs + sequence_tagging_obj_macs + global_corres_macs
total_params = rel_judgement_params + sequence_tagging_sub_params + sequence_tagging_obj_params + global_corres_params

print(f'**************************************************************')
print(f'Total MACs: {total_macs * (10 ** -6):.3f} M Params: {total_params}')
print(f'**************************************************************')
