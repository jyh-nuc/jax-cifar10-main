import jax
import jax.numpy as jnp
from flax import nnx
from model import CNN
from data import generate_synthetic_data, create_data_loader
from train import train

def main():
    print("开始生成随机数种子...")
    rngs = nnx.Rngs(0)
    
    print("生成合成数据...")
    train_data, val_data = generate_synthetic_data()
    print(f"训练集大小: {len(train_data[0])}, 验证集大小: {len(val_data[0])}")
    
    print("创建数据加载器...")
    train_loader = create_data_loader(train_data, batch_size=32)
    val_loader = create_data_loader(val_data, batch_size=32)
    
    print("初始化模型...")
    model = CNN(num_classes=10, rngs=rngs)
    
    print("开始训练 (10轮)...")
    trained_model = train(
        model, 
        train_loader, 
        val_loader, 
        epochs=10, 
        print_interval=1
    )
    
    print("开始测试...")
    test_data, _ = generate_synthetic_data(train=False)
    test_loader = create_data_loader(test_data, batch_size=32)
    accuracy = 0.0
    for x, y in test_loader:
        preds = trained_model(x)
        accuracy += jnp.mean(jnp.argmax(preds, axis=1) == y)
    accuracy /= len(test_loader)
    print(f"最终测试准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main()
