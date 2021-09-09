import os
import torch
from NetArchitecture import NetArchitecture
from NetDataset import NetDataset

class NetManager:
    def __init__(self):
        self.model = NetArchitecture()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        self.criterion = torch.nn.MSELoss()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        self.model.to(self.device)
        self.print_every = 50

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self, filename):
        state = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

    def set_device(self, device):
        if torch.cuda.is_available():
            self.device = torch.device(device)
        self.model.to(self.device)

    def test(self, test_input, test_output):
        test_dataset = NetDataset(test_input)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

        self.model.eval()
        torch.no_grad()
        count = len(test_loader)
        for i, (batch, _) in enumerate(test_loader):
            output = self.model(batch.to(self.device))
            test_dataset.save_output(output.to("cpu"), \
                    os.path.join(test_output, test_dataset.get_filename(i)))

