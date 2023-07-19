import os
import torch

from custom_utils.utils import (
    set_random_seeds,
    create_model,
    load_model,
    train_model,
    save_model,
    evaluate_model,
    create_classification_report,
    prepare_dataloader,
)


def main():
    random_seed = 0
    num_classes = 10
    l1_regularization_strength = 0
    l2_regularization_strength = 1e-4
    learning_rate = 1e-3
    num_epochs = 2
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "saved_models"
    model_filename = "resnet18_cifar10_new.pt"
    model_filepath = os.path.join(model_dir, model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = create_model(num_classes=num_classes)

    # Load a pretrained model.
    if os.path.exists(model_filepath):
        model = load_model(
            model=model, model_filepath=model_filepath, device=cuda_device
        )

    train_loader, test_loader, classes = prepare_dataloader(
        num_workers=4, train_batch_size=128, eval_batch_size=256
    )

    # Train model.
    print("Training Model...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=cuda_device,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    # Save model.
    save_model(model=model, model_dir=model_dir, model_filename=model_filename)

    _, eval_accuracy, _ = evaluate_model(
        model=model, test_loader=test_loader, device=cuda_device, criterion=None
    )

    classification_report = create_classification_report(
        model=model, test_loader=test_loader, device=cuda_device
    )

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Classification Report:")
    print(classification_report)


if __name__ == "__main__":
    main()
