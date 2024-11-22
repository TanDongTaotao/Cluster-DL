import torch

def test_cuda():
    """测试CUDA是否可用"""
    print("\n=== CUDA可用性测试 ===")
    
    # 检查CUDA是否可用
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # 显示CUDA信息
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        # 测试CUDA张量运算
        print("\n测试CUDA张量运算:")
        try:
            # 创建测试张量
            x = torch.rand(5, 3)
            print("CPU张量:")
            print(x)
            
            # 移动到GPU
            x = x.cuda()
            print("\nGPU张量:")
            print(x)
            
            print("\nCUDA张量运算测试成功!")
        except Exception as e:
            print(f"\nCUDA张量运算测试失败: {str(e)}")
    else:
        print("CUDA不可用，将使用CPU进行运算")
    
    print("="*20)

if __name__ == "__main__":
    test_cuda() 