﻿using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using StardewModdingAPI;
using StardewModdingAPI.Events;
using StardewValley;
using StardewValley.Minigames;
using static StardewValley.Minigames.AbigailGame;
using WindowsInput.Native;
using WindowsInput;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace swmod
{
    public class ModEntry : Mod
    {
        AbigailGame game = null;

        int lives = 3;
        const int SIZE = 24;

        int currentFrame = 0;
        int[] observations = new int[3072];
        bool shouldExecute = false;
        bool activePowerups = true;
        InputSimulator keyboard = new InputSimulator();
        List<VirtualKeyCode> pressedButtons = new List<VirtualKeyCode>();
        List<VirtualKeyCode> codes = new List<VirtualKeyCode>() { VirtualKeyCode.VK_W, VirtualKeyCode.VK_D, VirtualKeyCode.VK_S, VirtualKeyCode.VK_A };
        List<VirtualKeyCode> shots = new List<VirtualKeyCode>() { VirtualKeyCode.UP, VirtualKeyCode.RIGHT, VirtualKeyCode.DOWN, VirtualKeyCode.LEFT };

        List<CowboyMonster> lastMonsters = new List<CowboyMonster>();

        public override void Entry(IModHelper helper)
        {
            helper.Events.GameLoop.UpdateTicked += this.OnUpdateTicked;
            helper.Events.Input.ButtonPressed += this.OnButtonPressed;
            helper.ConsoleCommands.Add("move", "go in that way", Move);
        }

        private bool InMainGame() => Context.IsWorldReady;
        private bool InMiniGame()
        {
            if (Game1.currentMinigame != null && Game1.currentMinigame is AbigailGame game1)
            {
                if (game == null) game = game1;
                return true;
            }

            if (game != null) game = null;
            return  false;
        }

        private void UnpressAllButtons()
        {
            foreach (var button in pressedButtons)
                keyboard.Keyboard.KeyUp(button);
            pressedButtons.Clear();
        }

        private void Press(params VirtualKeyCode[] keys)
        {
            foreach (var key in keys)
            {
                keyboard.Keyboard.KeyDown(key);
                pressedButtons.Add(key);
            }
        }

        private void Move(string command, string[] args)
        {
            //Monitor.Log("do stuff: " + String.Join("", args), LogLevel.Debug);

            UnpressAllButtons();

            switch (args[0])
            {
                case "0":
                    break;
                case "1":
                    Press(codes[0]);
                    break;
                case "2":
                    Press(codes[0], codes[1]);
                    break;
                case "3":
                    Press(codes[1]);
                    break;
                case "4":
                    Press(codes[1], codes[2]);
                    break;
                case "5":
                    Press(codes[2]);
                    break;
                case "6":
                    Press(codes[2], codes[3]);
                    break;
                case "7":
                    Press(codes[3]);
                    break;
                case "8":
                    Press(codes[3], codes[0]);
                    break;
            }

            switch (args[1])
            {
                case "0":
                    Press(shots[0]);
                    break;
                case "1":
                    Press(shots[0], shots[1]);
                    break;
                case "2":
                    Press(shots[1]);
                    break;
                case "3":
                    Press(shots[1], shots[2]);
                    break;
                case "4":
                    Press(shots[2]);
                    break;
                case "5":
                    Press(shots[2], shots[3]);
                    break;
                case "6":
                    Press(shots[3]);
                    break;
                case "7":
                    Press(shots[3], shots[0]);
                    break;
            }

            // Monitor.Log(DateTime.Now.Millisecond + " end of move", LogLevel.Debug);
        }

        private Vector2 Centered(Vector2 playerPos)
        {
            return new Vector2(playerPos.X + 24, playerPos.Y + 16);
        }

        private void OnButtonPressed(object sender, ButtonPressedEventArgs e)
        {
            if (!InMainGame()) return;

            switch (e.Button)
            {
                case SButton.P:
                    shouldExecute = !shouldExecute;
                    break;
                case SButton.O:
                    activePowerups = !activePowerups;
                    break;
                case SButton.I:
                    Monitor.Log("observations", LogLevel.Debug);
                    break;
            }
        }

        private void OnUpdateTicked(object sender, EventArgs e)
        {
            if (!InMainGame() || !InMiniGame()) return;         // Nothing to do not in a mini-game.
            if (SkippedFrame()) return;                         // We want to send only 25% of frames (because nothing serious change in 1/60 of second).

            if (!ApplicationIsActivated() || !shouldExecute)
            {
                UnpressAllButtons();
                return;
            }

            if (!activePowerups) ClearPowerups();

            if (game.lives < lives)
            {
                lives = game.lives;
                UnpressAllButtons();
                return;
            }

            ClearObservations();
            SetPlayerPos();
            SetBulletPos();
            SetEnemyPos();

            Monitor.Log("data: " + string.Join(",", observations), LogLevel.Debug); // define current state
            // Monitor.Log(DateTime.Now.Millisecond + " end of update", LogLevel.Debug);
        }

        private bool SkippedFrame()
        {
            return (currentFrame++ % 10) != 0;
        }

        private void ClearPowerups()
        {
            if (powerups.Count > 0)
                powerups.Clear();
        }

        private void CastToNearestAngle(ref int x, ref int y)
        {
            int bestX = (x / SIZE) * SIZE, bestY = (y / SIZE) * SIZE;

            if (x % SIZE >= SIZE / 2) bestX += SIZE;
            if (y % SIZE >= SIZE / 2) bestY += SIZE;

            x = bestX;
            y = bestY;
        }

        private void ClearObservations()
        {
            for (int i = 0; i < observations.Length; ++i)
                observations[i] = 0;
        }

        private void SetPlayerPos()
        {
            var playerCenter = Centered(game.playerPosition);
            int x = (int)playerCenter.X;
            int y = (int)playerCenter.Y;
            CastToNearestAngle(ref x, ref y);
            x /= SIZE;
            y /= SIZE;

            observations[x + 32 * y] = 1;
            observations[x + 1 + 32 * y] = 1;
            observations[x + 1 + 32 * (y + 1)] = 1;
            observations[x + 32 * (y + 1)] = 1;
        }

        private void SetBulletPos()
        {
            foreach(var bullet in game.bullets)
            {
                var bulletCenter = bullet.position;
                int x = bulletCenter.X / 24;
                int y = bulletCenter.Y / 24;

                observations[1024 + x + 32 * y] = 1;
            }
        }

        private void SetEnemyPos()
        {
            foreach (CowboyMonster monster in monsters)
            {

                var monsterCenter = monster.position.Center;
                int x = monsterCenter.X;
                int y = monsterCenter.Y;

                CastToNearestAngle(ref x, ref y);
                x /= SIZE;
                y /= SIZE;

                observations[2048 + x + 32 * y] = 1;
                observations[2048 + x + 1 + 32 * y] = 1;
                observations[2048 + x + 1 + 32 * (y + 1)] = 1;
                observations[2048 + x + 32 * (y + 1)] = 1;
            }
        }

        /// <summary>
        /// Returns true if the current application has focus, false otherwise
        /// </summary>
        public static bool ApplicationIsActivated()
        {
            var activatedHandle = GetForegroundWindow();
            if (activatedHandle == IntPtr.Zero)
            {
                return false;       // No window is currently activated
            }

            var procId = Process.GetCurrentProcess().Id;
            int activeProcId;
            GetWindowThreadProcessId(activatedHandle, out activeProcId);

            return activeProcId == procId;
        }

        [DllImport("user32.dll", CharSet = CharSet.Auto, ExactSpelling = true)]
        private static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern int GetWindowThreadProcessId(IntPtr handle, out int processId);
    }
}