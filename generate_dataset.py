"""
Synthetic Signal Dataset Generator for Modulation Classification
Generates synthetic wireless signals with various modulation schemes and SNR levels
"""

import numpy as np
import pickle
from typing import Tuple, List, Dict
import os

class SignalGenerator:
    """Generate synthetic modulated signals with various modulation schemes"""
    
    def __init__(self, samples_per_symbol: int = 8, num_samples: int = 128):
        self.samples_per_symbol = samples_per_symbol
        self.num_samples = num_samples
        self.modulation_types = [
            'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM',
            'AM-DSB', 'AM-SSB', 'FM', 'GFSK', 'CPFSK', 'PAM4'
        ]
        
    def generate_bpsk(self, num_symbols: int) -> np.ndarray:
        """Generate BPSK modulated signal"""
        bits = np.random.randint(0, 2, num_symbols)
        symbols = 2 * bits - 1  # Map to {-1, 1}
        signal = np.repeat(symbols, self.samples_per_symbol)
        return signal[:self.num_samples] + 1j * np.zeros(self.num_samples)
    
    def generate_qpsk(self, num_symbols: int) -> np.ndarray:
        """Generate QPSK modulated signal"""
        bits_i = np.random.randint(0, 2, num_symbols)
        bits_q = np.random.randint(0, 2, num_symbols)
        symbols_i = 2 * bits_i - 1
        symbols_q = 2 * bits_q - 1
        symbols = (symbols_i + 1j * symbols_q) / np.sqrt(2)
        signal = np.repeat(symbols, self.samples_per_symbol)
        return signal[:self.num_samples]
    
    def generate_8psk(self, num_symbols: int) -> np.ndarray:
        """Generate 8PSK modulated signal"""
        phases = np.random.randint(0, 8, num_symbols)
        symbols = np.exp(1j * 2 * np.pi * phases / 8)
        signal = np.repeat(symbols, self.samples_per_symbol)
        return signal[:self.num_samples]
    
    def generate_16qam(self, num_symbols: int) -> np.ndarray:
        """Generate 16QAM modulated signal"""
        constellation = []
        for i in [-3, -1, 1, 3]:
            for q in [-3, -1, 1, 3]:
                constellation.append(i + 1j * q)
        constellation = np.array(constellation) / np.sqrt(10)
        
        indices = np.random.randint(0, 16, num_symbols)
        symbols = constellation[indices]
        signal = np.repeat(symbols, self.samples_per_symbol)
        return signal[:self.num_samples]
    
    def generate_64qam(self, num_symbols: int) -> np.ndarray:
        """Generate 64QAM modulated signal"""
        constellation = []
        for i in range(-7, 8, 2):
            for q in range(-7, 8, 2):
                constellation.append(i + 1j * q)
        constellation = np.array(constellation) / np.sqrt(42)
        
        indices = np.random.randint(0, 64, num_symbols)
        symbols = constellation[indices]
        signal = np.repeat(symbols, self.samples_per_symbol)
        return signal[:self.num_samples]
    
    def generate_am_dsb(self, num_symbols: int) -> np.ndarray:
        """Generate AM-DSB modulated signal"""
        t = np.linspace(0, 1, self.num_samples)
        message = np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2*np.pi))
        carrier_freq = 20
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        signal = (1 + 0.5 * message) * carrier
        return signal + 1j * np.zeros(self.num_samples)
    
    def generate_am_ssb(self, num_symbols: int) -> np.ndarray:
        """Generate AM-SSB modulated signal"""
        t = np.linspace(0, 1, self.num_samples)
        message = np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2*np.pi))
        carrier_freq = 20
        signal = message * np.exp(1j * 2 * np.pi * carrier_freq * t)
        return signal
    
    def generate_fm(self, num_symbols: int) -> np.ndarray:
        """Generate FM modulated signal"""
        t = np.linspace(0, 1, self.num_samples)
        message = np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2*np.pi))
        carrier_freq = 20
        kf = 10  # Frequency deviation constant
        phase = 2 * np.pi * carrier_freq * t + 2 * np.pi * kf * np.cumsum(message) / self.num_samples
        signal = np.exp(1j * phase)
        return signal
    
    def generate_gfsk(self, num_symbols: int) -> np.ndarray:
        """Generate GFSK modulated signal"""
        bits = np.random.randint(0, 2, num_symbols)
        bits = 2 * bits - 1  # Map to {-1, 1}
        
        # Gaussian filter
        bt = 0.3
        span = 4
        sps = self.samples_per_symbol
        t = np.arange(-span*sps/2, span*sps/2) / sps
        h = np.exp(-2 * np.pi**2 * bt**2 * t**2)
        h = h / np.sum(h)
        
        # Upsample and filter
        upsampled = np.zeros(num_symbols * sps)
        upsampled[::sps] = bits
        filtered = np.convolve(upsampled, h, mode='same')
        
        # Frequency modulation
        kf = 0.5
        phase = 2 * np.pi * kf * np.cumsum(filtered)
        signal = np.exp(1j * phase)
        return signal[:self.num_samples]
    
    def generate_cpfsk(self, num_symbols: int) -> np.ndarray:
        """Generate CPFSK modulated signal"""
        bits = np.random.randint(0, 2, num_symbols)
        bits = 2 * bits - 1  # Map to {-1, 1}
        upsampled = np.repeat(bits, self.samples_per_symbol)
        
        kf = 0.5
        phase = 2 * np.pi * kf * np.cumsum(upsampled)
        signal = np.exp(1j * phase)
        return signal[:self.num_samples]
    
    def generate_pam4(self, num_symbols: int) -> np.ndarray:
        """Generate PAM4 modulated signal"""
        symbols = np.random.choice([-3, -1, 1, 3], num_symbols) / np.sqrt(5)
        signal = np.repeat(symbols, self.samples_per_symbol)
        return signal[:self.num_samples] + 1j * np.zeros(self.num_samples)
    
    def add_awgn(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Additive White Gaussian Noise to signal"""
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )
        return signal + noise
    
    def generate_signal(self, modulation: str, snr_db: float) -> np.ndarray:
        """Generate a signal with specified modulation and SNR"""
        num_symbols = self.num_samples // self.samples_per_symbol + 1
        
        if modulation == 'BPSK':
            signal = self.generate_bpsk(num_symbols)
        elif modulation == 'QPSK':
            signal = self.generate_qpsk(num_symbols)
        elif modulation == '8PSK':
            signal = self.generate_8psk(num_symbols)
        elif modulation == '16QAM':
            signal = self.generate_16qam(num_symbols)
        elif modulation == '64QAM':
            signal = self.generate_64qam(num_symbols)
        elif modulation == 'AM-DSB':
            signal = self.generate_am_dsb(num_symbols)
        elif modulation == 'AM-SSB':
            signal = self.generate_am_ssb(num_symbols)
        elif modulation == 'FM':
            signal = self.generate_fm(num_symbols)
        elif modulation == 'GFSK':
            signal = self.generate_gfsk(num_symbols)
        elif modulation == 'CPFSK':
            signal = self.generate_cpfsk(num_symbols)
        elif modulation == 'PAM4':
            signal = self.generate_pam4(num_symbols)
        else:
            raise ValueError(f"Unknown modulation type: {modulation}")
        
        # Add noise
        noisy_signal = self.add_awgn(signal, snr_db)
        
        # Normalize
        noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
        
        return noisy_signal


def generate_dataset(
    num_samples_per_mod: int = 1000,
    snr_range: Tuple[int, int] = (-20, 20),
    snr_step: int = 2,
    num_samples: int = 128,
    output_file: str = 'dataset.pkl'
) -> Dict:
    """
    Generate complete dataset with multiple modulation types and SNR levels
    
    Args:
        num_samples_per_mod: Number of samples per modulation type per SNR
        snr_range: Range of SNR values (min, max) in dB
        snr_step: Step size for SNR values
        num_samples: Number of samples per signal
        output_file: Output pickle file name
    
    Returns:
        Dictionary containing dataset information
    """
    generator = SignalGenerator(num_samples=num_samples)
    snr_values = list(range(snr_range[0], snr_range[1] + 1, snr_step))
    
    X = []  # Signals
    y = []  # Labels (modulation type)
    snr_labels = []  # SNR values
    
    mod_to_idx = {mod: idx for idx, mod in enumerate(generator.modulation_types)}
    
    print(f"Generating dataset with {len(generator.modulation_types)} modulation types")
    print(f"SNR range: {snr_range[0]} to {snr_range[1]} dB (step: {snr_step})")
    print(f"Samples per modulation per SNR: {num_samples_per_mod}")
    
    total_samples = len(generator.modulation_types) * len(snr_values) * num_samples_per_mod
    print(f"Total samples to generate: {total_samples}")
    
    sample_count = 0
    for mod_type in generator.modulation_types:
        for snr in snr_values:
            for _ in range(num_samples_per_mod):
                signal = generator.generate_signal(mod_type, snr)
                
                # Store as [2, num_samples] array (I and Q components)
                signal_iq = np.vstack([signal.real, signal.imag])
                
                X.append(signal_iq)
                y.append(mod_to_idx[mod_type])
                snr_labels.append(snr)
                
                sample_count += 1
                if sample_count % 10000 == 0:
                    print(f"Generated {sample_count}/{total_samples} samples...")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    snr_labels = np.array(snr_labels, dtype=np.float32)
    
    # Shuffle dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    snr_labels = snr_labels[indices]
    
    dataset = {
        'X': X,
        'y': y,
        'snr': snr_labels,
        'modulations': generator.modulation_types,
        'mod_to_idx': mod_to_idx,
        'idx_to_mod': {idx: mod for mod, idx in mod_to_idx.items()}
    }
    
    # Save dataset
    output_path = os.path.join('/home/ubuntu/signal_classification_project', output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nDataset generated successfully!")
    print(f"Shape: X={X.shape}, y={y.shape}, snr={snr_labels.shape}")
    print(f"Saved to: {output_path}")
    
    return dataset


if __name__ == "__main__":
    # Generate dataset
    dataset = generate_dataset(
        num_samples_per_mod=1000,
        snr_range=(-20, 20),
        snr_step=2,
        num_samples=128,
        output_file='signal_dataset.pkl'
    )
    
    print("\nDataset Statistics:")
    print(f"Number of modulation types: {len(dataset['modulations'])}")
    print(f"Modulation types: {dataset['modulations']}")
    print(f"Total samples: {len(dataset['X'])}")
    print(f"Signal shape: {dataset['X'][0].shape}")
    print(f"SNR range: [{dataset['snr'].min():.1f}, {dataset['snr'].max():.1f}] dB")

