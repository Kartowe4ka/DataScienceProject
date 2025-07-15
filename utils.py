import time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return total_loss / len(data_loader), correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, test_loader, epochs=100, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    train_losses_plt, train_accs_plt = [], []
    test_losses_plt, test_accs_plt = [], []

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        train_losses_plt.append(train_loss)
        train_accs_plt.append(train_acc)
        test_losses_plt.append(test_loss)
        test_accs_plt.append(test_acc)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))

    ax1.plot(history['train_losses'], label="Train Loss")
    ax1.plot(history['test_losses'], label="Test Loss")
    ax1.legend()

    ax2.plot(history['train_accs'], label="Train Acc")
    ax2.plot(history['test_accs'], label="Test Acc")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def get_experiment_data(model, train_loader, test_loader, device, name):
    # Обучение
    start_train = time.time()
    history = train_model(model, train_loader, test_loader, epochs=100, device=str(device))
    end_train = time.time()
    train = end_train - start_train

    # Инференс
    model.eval()
    start_infer = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
    end_infer = time.time()
    infer = end_infer - start_infer

    # Подсчёт параметров
    params = count_parameters(model)

    print(f"{name}: Кол-во параметров - {params}\nВремя обучения - {train / 60:.2f} минут \nВремя инференса - {infer:.2f} секунд")
    plot_training_history(history)
