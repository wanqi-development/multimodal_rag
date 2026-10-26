@echo off
REM ?????????????????? frontend.ps1 ?? ?? ???????
REM Why the wrapper: see the note in backend.cmd (execution policy on unzipped .ps1).
REM NOTE: keep this file ASCII-only.
REM Usage: scripts\windows\frontend.cmd             or   ... -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0frontend.ps1" %*
