
using System.Reflection;

using usearch_index_t = System.IntPtr;
using usearch_key_t = System.UInt64;
using usearch_distance_t = System.Single;
using usearch_error_t = System.IntPtr;
using size_t = System.UIntPtr;
using void_ptr_t = System.IntPtr;

internal static class NativeMethods
{
    private const string LibraryName = "libusearch_c";

    #region Resolving library path
    static NativeMethods()
    {
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, ImportResolver);
    }

    private static IntPtr ImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == LibraryName)
        {
            string path = $"./runtimes/{GetRuntimeIdentifier()}/native/libusearch_c{GetLibraryExtension()}";
            if (NativeLibrary.TryLoad(path, out IntPtr libHandle))
            {
                return libHandle;
            }
        }
        return IntPtr.Zero;
    }

    private static string GetRuntimeIdentifier()
        => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? (Environment.Is64BitProcess ? "win10-x64" : "win10-x86")
         : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? (Environment.Is64BitProcess ? "linux-x64" : "linux-x86")
         : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? (Environment.Is64BitProcess ? "osx-x64" : "osx-x86")
         : throw new PlatformNotSupportedException("Unsupported platform");

    private static string GetLibraryExtension()
        => RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? ".dll"
         : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? ".so"
         : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? ".dylib"
         : throw new PlatformNotSupportedException("Unsupported platform");

    public class PlatformNotSupportedException : Exception { public PlatformNotSupportedException(string message) : base(message) { } }
    #endregion


    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern usearch_index_t usearch_init(ref usearch_init_options_t options, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_free(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_save(usearch_index_t index, [MarshalAs(UnmanagedType.LPStr)] string path, out usearch_error_t error);

    [DllImport(LibraryName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_load(usearch_index_t index, [MarshalAs(UnmanagedType.LPStr)] string path, out usearch_error_t error);

    [DllImport(LibraryName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_view(usearch_index_t index, [MarshalAs(UnmanagedType.LPStr)] string path, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint usearch_size(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint usearch_capacity(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint usearch_dimensions(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint usearch_connectivity(usearch_index_t index, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_reserve(usearch_index_t index, size_t capacity, out usearch_error_t error);


    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_add(
        usearch_index_t index,
        usearch_key_t key,
        [In] Half[] vector,
        usearch_scalar_kind_t vector_kind,
        out usearch_error_t error
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_add(
        usearch_index_t index,
        usearch_key_t key,
        [In] float[] vector,
        usearch_scalar_kind_t vector_kind,
        out usearch_error_t error
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void usearch_add(
        usearch_index_t index,
        usearch_key_t key,
        [In] double[] vector,
        usearch_scalar_kind_t vector_kind,
        out usearch_error_t error
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool usearch_contains(usearch_index_t index, usearch_key_t key, out usearch_error_t error);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint usearch_search(
        usearch_index_t index,
        IntPtr query_vector,
        usearch_scalar_kind_t query_kind,
        size_t results_limit,
        [Out] usearch_key_t[] found_keys,
        [Out] usearch_distance_t[] found_distances,
        out usearch_error_t error
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint usearch_search(
        usearch_index_t index,
        [In] Half[] query_vector,
        usearch_scalar_kind_t query_kind,
        size_t results_limit,
        [Out] usearch_key_t[] found_keys,
        [Out] usearch_distance_t[] found_distances,
        out usearch_error_t error
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint usearch_search(
        usearch_index_t index,
        [In] float[] query_vector,
        usearch_scalar_kind_t query_kind,
        size_t results_limit,
        [Out] usearch_key_t[] found_keys,
        [Out] usearch_distance_t[] found_distances,
        out usearch_error_t error
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint usearch_search(
        usearch_index_t index,
        [In] double[] query_vector,
        usearch_scalar_kind_t query_kind,
        size_t results_limit,
        [Out] usearch_key_t[] found_keys,
        [Out] usearch_distance_t[] found_distances,
        out usearch_error_t error
    );


    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool usearch_get(
         usearch_index_t index,
         usearch_key_t key,
         [Out] Half[] vector,
         usearch_scalar_kind_t vector_kind,
         out usearch_error_t error
     );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool usearch_get(
        usearch_index_t index,
        usearch_key_t key,
        [Out] float[] vector,
        usearch_scalar_kind_t vector_kind,
        out usearch_error_t error
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool usearch_get(
        usearch_index_t index,
        usearch_key_t key,
        [Out] double[] vector,
        usearch_scalar_kind_t vector_kind,
        out usearch_error_t error
    );

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public static extern bool usearch_remove(usearch_index_t index, usearch_key_t key, out usearch_error_t error);
}
