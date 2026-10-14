@echo off
REM ????????? backend.ps1 ?? ?? ???????
REM Why the wrapper: Windows PowerShell refuses to run .ps1 files that were unzipped
REM from an archive (they carry the "downloaded from the internet" mark) unless the
REM execution policy allows it. This passes -ExecutionPolicy Bypass for this single
REM call only -- it does NOT change any machine setting.
REM NOTE: keep this file ASCII-only. cmd.exe reads .bat/.cmd in the OEM codepage,
REM so non-ASCII comments here get executed as garbage commands.
REM Usage: scripts\windows\backend.cmd            or   ... -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend.ps1" %*
