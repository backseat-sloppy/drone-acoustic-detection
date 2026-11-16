@echo off
rem Local preview of the GitHub Pages site. Double-click to start.
rem
rem The first time on a machine it installs what it needs (a few minutes):
rem   1. Ruby 3.3 with its build tools (through winget, Windows' package manager)
rem   2. the gems in Gemfile.lock (Jekyll and the GitHub Pages plugins)
rem After that it starts the server, opens the site in your browser once it is
rem built, and reloads the page every time you save a file.
rem Close this window (or press Ctrl+C) to stop the server.

setlocal
title Docs site preview
cd /d "%~dp0"

set "RUBY_DIR=C:\Ruby33-x64\bin"

rem --- 1. Ruby -----------------------------------------------------------------
where ruby >nul 2>nul && goto have_ruby
if exist "%RUBY_DIR%\ruby.exe" goto add_ruby

echo Ruby is not installed. Installing Ruby 3.3 with its build tools...
echo Windows may ask for permission. This takes a few minutes.
where winget >nul 2>nul || (
    echo.
    echo winget was not found. Install "App Installer" from the Microsoft Store,
    echo or install Ruby by hand from https://rubyinstaller.org ^(Ruby+Devkit 3.3, x64^).
    goto fail
)
winget install --id RubyInstallerTeam.RubyWithDevKit.3.3 -e --silent --accept-source-agreements --accept-package-agreements
if not exist "%RUBY_DIR%\ruby.exe" (
    echo.
    echo Ruby did not end up in %RUBY_DIR%. If it was installed somewhere else,
    echo open a new window and run this script again.
    goto fail
)

:add_ruby
set "PATH=%RUBY_DIR%;%PATH%"

:have_ruby
rem The build tools (MSYS2 with gcc) that some gems need to compile.
rem They sit next to Ruby; install them only if the compiler is missing.
for %%i in (ruby.exe) do set "RUBY_BIN=%%~dp$PATH:i"
if not exist "%RUBY_BIN%..\msys64\ucrt64\bin\gcc.exe" (
    echo Installing Ruby's build tools. This only happens once...
    call ridk install 3 || goto fail
)

rem --- 2. Gems -----------------------------------------------------------------
call bundle check >nul 2>nul || (
    echo Installing the site's gems. This only happens once...
    call bundle install || goto fail
)

rem --- 3. Serve ----------------------------------------------------------------
rem A server left over from another window (or a terminal tab) holds port 4000,
rem and would keep serving the site after this window is closed.
set "OLD_PID="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":4000 .*LISTENING"') do set "OLD_PID=%%p"
if not defined OLD_PID goto serve
echo.
echo A site server is already running ^(process %OLD_PID%^), probably in another
echo window or terminal tab.
set "ANSWER="
set /p "ANSWER=Stop it and start a fresh one here? [Y/N] "
if /i not "%ANSWER%"=="Y" (
    echo Leaving it running and opening the site.
    start "" "http://localhost:4000/drone-acoustic-detection/"
    exit /b 0
)
taskkill /PID %OLD_PID% /T /F >nul
ping -n 3 127.0.0.1 >nul

:serve
echo.
echo Starting the site. Your browser opens when it is built.
echo Close this window to stop it.
echo.
call bundle exec jekyll serve --livereload --open-url
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo Something went wrong; see the messages above.
pause
exit /b 1
