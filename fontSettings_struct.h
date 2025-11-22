#pragma once

struct FontSettings
{
    int family;
    bool isBold;
    bool isItalic;
    int size;

    FontSettings()
    {
        family = 0;
        isBold = false;
        isItalic = false;
        size = 24;
    }

    FontSettings(int _family, bool _isBold, bool _isItalic, int _size)
    {
        family = _family;
        isBold = _isBold;
        isItalic = _isItalic;
        size = _size;
    }
};