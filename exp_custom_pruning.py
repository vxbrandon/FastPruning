import torch
import os

from custom_utils.pruning.diag_pruning import diag_pruning_linear
from custom_utils.utils import prepare_dataloader, evaluate_model, train_model, load_model, set_random_seeds
from models.fcn import FCN

def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv

def generate_perm_tensor(image_size: int = 32, block_size: int = 4):
  assert image_size % block_size == 0
  perm_list = []
  for row in range(image_size):
    for col in range(image_size):
      i = int(row/block_size)
      j = int(col/block_size)
      previous_block_nums = i * image_size * block_size + j * block_size * block_size

      i = row % block_size
      j = col % block_size
      index = previous_block_nums + i * block_size + j
      perm_list.append(index)
  return inverse_permutation(torch.tensor(perm_list))

import torch
import torch.nn as nn
import scipy.sparse as sparse
import betterspy

def main():
  seed = 999
  set_random_seeds(seed)
  cuda_device = torch.device("cuda:0")
  data_type = "CIFAR_10"

  # Experimental setting
  lr_rate = 1e-2
  num_epochs=300
  patience=10
  verbose = True

  # Load dataset
  train_loader, test_loader, classes = prepare_dataloader(
          num_workers=8,
          train_batch_size=128,
          eval_batch_size=256,
          data_type=data_type,
          is_flatten=True,)
  input_dims = next(iter(train_loader))[0].size()[-1]
  num_layers = 5
  fcn = FCN(num_layers=num_layers, input_dims=input_dims)

  model_filename = f"FCN_{num_layers}_{data_type}.pt"
  model_dir = "saved_models"
  model_filepath = os.path.join(model_dir, model_filename)
  if not (os.path.exists(model_dir)):
    os.makedirs(model_dir)

  # fcn = utils.load_model(fcn, model_filepath, cuda_device)
  # _, eval_accuracy, _ = utils.evaluate_model(model=fcn,
  #                                         test_loader=test_loader,
  #                                         device=cuda_device,
  #                                         criterion=None)
  # print(f"Before permutation: Test Accuracy {eval_accuracy}")

  # fcn = utils.FCN(num_layers=num_layers, input_dims=input_dims)
  counts = 0
  for module in fcn.modules():
      if isinstance(module, nn.Linear):
          a = generate_perm_tensor(image_size=32, block_size=4)
          perm_tensor = torch.cat((a, a + 32*32, a + 32*32*2))
          print(perm_tensor)
          diag_pruning_linear(module, block_size=16, perm_type="CUSTOM", row_perm=perm_tensor, col_perm=perm_tensor)
          counts += 1
      if counts >=2:
          break
  sparse_matrix = sparse.csr_matrix(fcn.fcn.fc0.weight.cpu().detach().numpy())

  # Show and save the sparsity pattern
  betterspy.show(sparse_matrix)
  train_model(model=fcn,
                          train_loader=train_loader,
                          test_loader=test_loader,
                          device=cuda_device,
                          learning_rate=lr_rate,
                          num_epochs=num_epochs,
                          T_max=num_epochs,
                          patience=patience,
                          verbose=verbose,
                          optimizer="SGD")

  _, eval_accuracy, _ = evaluate_model(model=fcn,
                                          test_loader=test_loader,
                                          device=cuda_device,
                                          criterion=None)
  print(f"After permutation: Test Accuracy {eval_accuracy}")
main()