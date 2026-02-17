#include "commonItemDialogs.h"

COMDLG_FILTERSPEC fileType = { L"CUDAynamics System Configuration", L"*.cfg" };

bool CommonItemDialogSave(std::string& content)
{
	// Selected kernel name as default .cfg-file name
	std::wstring selKernelW = std::wstring(selectedKernel.begin(), selectedKernel.end());
	LPCWSTR selKernelL = selKernelW.c_str();

	HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
	if (FAILED(hr)) return false;

	IFileSaveDialog* pDialog = NULL;
	hr = CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_ALL, IID_PPV_ARGS(&pDialog));
	if (FAILED(hr)) goto Uninitialize;

	pDialog->SetFileTypes(1, &fileType);
	pDialog->SetDefaultExtension(L"cfg");
	pDialog->SetFileName(selKernelL);
	hr = pDialog->Show(guiHwnd);
	if (FAILED(hr)) goto DialogRelease;

	IShellItem* pItem;
	hr = pDialog->GetResult(&pItem);
	if (FAILED(hr)) goto DialogRelease;

	PWSTR pszPath;
	hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszPath);
	if (FAILED(hr)) goto ItemRelease;

	{
		char path[MAX_PATH];
		WideCharToMultiByte(CP_UTF8, 0, pszPath, -1, path, MAX_PATH, NULL, NULL);
		std::ofstream file(path);
		if (file.is_open())
		{
			file << content;
			file.close();
			//MessageBoxA(guiHwnd, "Configuration saved", "Success", MB_OK);
		}

		CoTaskMemFree(pszPath);
	}

ItemRelease:
	pItem->Release();
DialogRelease:
	pDialog->Release();
Uninitialize:
	CoUninitialize();
	return SUCCEEDED(hr);
}

bool CommonItemDialogLoad(std::string* content, bool* userCancelled)
{
	std::string readContent = "";
	*userCancelled = false;

	HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
	if (FAILED(hr)) return false;

	IFileSaveDialog* pDialog = NULL;
	hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_INPROC_SERVER, IID_IFileOpenDialog, (void**)&pDialog);
	if (FAILED(hr)) goto Uninitialize;

	pDialog->SetFileTypes(1, &fileType);
	pDialog->SetDefaultExtension(L"cfg");
	DWORD flags;
	pDialog->GetOptions(&flags);
	pDialog->SetOptions(flags | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST);
	hr = pDialog->Show(guiHwnd);

	if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED))
	{
		*userCancelled = true;
		goto DialogRelease;
	}

	if (FAILED(hr)) goto DialogRelease;

	IShellItem* pItem;
	hr = pDialog->GetResult(&pItem);
	if (FAILED(hr)) goto DialogRelease;

	PWSTR pszPath;
	hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszPath);
	if (FAILED(hr)) goto ItemRelease;

	{
		char path[MAX_PATH];
		WideCharToMultiByte(CP_UTF8, 0, pszPath, -1, path, MAX_PATH, NULL, NULL);
		std::ifstream file(path);
		if (file.is_open())
		{
			for (std::string line; getline(file, line); ) readContent += line + '\n';
			file.close();
		}

		*content = readContent;
		CoTaskMemFree(pszPath);
	}

ItemRelease:
	pItem->Release();
DialogRelease:
	pDialog->Release();
Uninitialize:
	CoUninitialize();

	return SUCCEEDED(hr);
}