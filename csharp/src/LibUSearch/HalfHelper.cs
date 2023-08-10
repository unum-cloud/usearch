public static class HalfHelper
{
    public static ushort Pack(Half value)
    {
        return (ushort)BitConverter.ToInt16(BitConverter.GetBytes(value), 0);
    }
    public static Half UnPack(short value)
    {
        return BitConverter.ToHalf(BitConverter.GetBytes(value), 0);
    }
}