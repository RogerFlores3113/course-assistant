# Course Assistant installer for Windows 10/11 (64-bit), Windows PowerShell 5.1 or newer.
#
#   irm https://raw.githubusercontent.com/RogerFlores3113/course-assistant/main/install.ps1 | iex
#
# Options:  -NoOpen   install but do not open the app
# Testing:  $env:COURSE_ASSISTANT_INSTALL_FROM = folder or base URL holding the release files
#           $env:COURSE_ASSISTANT_INSTALL_DIR  = folder to install into
param([switch]$NoOpen)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }

$BaseUrl = 'https://github.com/RogerFlores3113/course-assistant/releases/latest/download'
$Asset = 'CourseAssistant-windows.zip'

function Stop-Install([string]$Message) {
    Write-Host ''
    Write-Host 'Course Assistant was not installed.' -ForegroundColor Red
    Write-Host $Message
    throw 'CA_INSTALL_STOPPED'
}

function Install-CourseAssistant {
    $Source = $env:COURSE_ASSISTANT_INSTALL_FROM
    if (-not $Source) { $Source = $BaseUrl }
    $Override = $env:COURSE_ASSISTANT_INSTALL_DIR

    if (-not [Environment]::Is64BitOperatingSystem) {
        Write-Host ''
        Write-Host 'Course Assistant was not installed.' -ForegroundColor Red
        Write-Host 'Course Assistant is only built for 64-bit Windows.'
        $global:LASTEXITCODE = 1
        return
    }

    if ($Override) { $Target = $Override } else { $Target = Join-Path $env:LOCALAPPDATA 'Programs\Course Assistant' }

    $Work = Join-Path ([IO.Path]::GetTempPath()) ('course-assistant-install-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $Work | Out-Null

    try {
        function Get-ReleaseFile([string]$Name, [string]$Dest) {
            if ($Source -match '^https?://') {
                Invoke-WebRequest -UseBasicParsing -Uri ($Source.TrimEnd('/') + '/' + $Name) -OutFile $Dest
            } else {
                $local = Join-Path $Source $Name
                if (-not (Test-Path -LiteralPath $local)) { throw "Missing $local" }
                Copy-Item -LiteralPath $local -Destination $Dest
            }
        }

        Write-Host 'Installing Course Assistant...'
        Write-Host "Downloading $Asset"
        $zip = Join-Path $Work $Asset
        $sums = Join-Path $Work 'SHA256SUMS.txt'
        try { Get-ReleaseFile $Asset $zip } catch { Stop-Install "Could not download $Asset. Check your internet connection and try again." }
        try { Get-ReleaseFile 'SHA256SUMS.txt' $sums } catch { Stop-Install 'Could not download SHA256SUMS.txt, so the download cannot be checked. Check your internet connection and try again.' }

        Write-Host 'Checking the download'
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLower()
        $expected = $null
        foreach ($line in Get-Content -LiteralPath $sums) {
            $parts = $line.Trim() -split '\s+', 2
            if ($parts.Count -eq 2 -and $parts[1].TrimStart('*') -eq $Asset) { $expected = $parts[0].ToLower(); break }
        }
        if (-not $expected) { Stop-Install "SHA256SUMS.txt does not list $Asset, so the download cannot be checked." }
        if ($actual -ne $expected) {
            Stop-Install "The download does not match its checksum, so it was not installed. Expected $expected but got $actual. Try again later, and tell the Course Assistant maintainer if it keeps happening."
        }
        Write-Host 'Download verified'

        $unpacked = Join-Path $Work 'unpacked'
        Expand-Archive -LiteralPath $zip -DestinationPath $unpacked -Force
        $inner = Join-Path $unpacked 'Course Assistant'
        if (-not (Test-Path -LiteralPath (Join-Path $inner 'Course Assistant.exe'))) {
            Stop-Install 'The download did not contain Course Assistant.exe.'
        }

        $upgrade = $false
        # Stop a running copy that lives in the install folder.
        if (Test-Path -LiteralPath $Target) {
            $full = [IO.Path]::GetFullPath($Target).TrimEnd('\') + '\'
            $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
                $_.ProcessName -in @('Course Assistant', 'course-assistant-mcp') -and $_.Path -and $_.Path.StartsWith($full, [StringComparison]::OrdinalIgnoreCase)
            })
            if ($running.Count -gt 0) {
                Write-Host 'Closing Course Assistant so it can be updated. Your courses and settings are stored elsewhere and are kept.'
                $running | Stop-Process -Force
                Start-Sleep -Seconds 2
            }
            $upgrade = $true
            Write-Host 'Replacing the existing copy. Your courses and settings are kept.'
            try { Remove-Item -LiteralPath $Target -Recurse -Force } catch {
                Stop-Install "Could not replace $Target. Please close Course Assistant and try again. Your data is safe."
            }
        }

        New-Item -ItemType Directory -Path $Target -Force | Out-Null
        Copy-Item -Path (Join-Path $inner '*') -Destination $Target -Recurse -Force
        $exe = Join-Path $Target 'Course Assistant.exe'
        Write-Host "Installed to $Target"

        if (-not $Override) {
            try {
                $programs = Join-Path ([Environment]::GetFolderPath('Programs')) 'Course Assistant.lnk'
                $shell = New-Object -ComObject WScript.Shell
                $lnk = $shell.CreateShortcut($programs)
                $lnk.TargetPath = $exe
                $lnk.WorkingDirectory = $Target
                $lnk.Save()
                Write-Host 'Added Course Assistant to your Start menu'
            } catch {
                Write-Host 'Could not add a Start menu shortcut. You can still open Course Assistant from its folder.'
            }
        }

        Write-Host ''
        Write-Host 'Done. Your courses and settings live in their own folder and are never touched by installs or updates.'
        Write-Host 'Course Assistant is in your Start menu.'
        $cfg = Join-Path $env:APPDATA 'Claude\claude_desktop_config.json'
        $connected = $upgrade -and (Test-Path -LiteralPath $cfg) -and (Select-String -LiteralPath $cfg -SimpleMatch 'course-assistant' -Quiet)
        if (-not $NoOpen) {
            Write-Host 'Opening Course Assistant'
            Start-Process -FilePath $exe -WorkingDirectory $Target
            Write-Host 'Course Assistant should now be open in your web browser.'
        }
        if ($connected) {
            Write-Host 'Already connected. Restart Claude Desktop to use the new version.'
        } elseif (-not $NoOpen) {
            Write-Host 'Next: to use Claude Desktop or Codex with your courses, open the Connect page in the app.'
        } else {
            Write-Host 'Next: to use Claude Desktop or Codex with your courses, open Course Assistant and go to its Connect page.'
        }
        Write-Host 'Connecting Claude Desktop needs it to be restarted, so save anything you are typing there first.'
    }
    catch {
        if ($_.Exception.Message -ne 'CA_INSTALL_STOPPED') {
            Write-Host ''
            Write-Host 'Course Assistant was not installed.' -ForegroundColor Red
            Write-Host $_.Exception.Message
        }
        $global:LASTEXITCODE = 1
        return
    }
    finally {
        Remove-Item -LiteralPath $Work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Install-CourseAssistant
