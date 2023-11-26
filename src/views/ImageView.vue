<script setup>
import ChatWindowImage from '../components/ChatWindowImage.vue'
import { useImageChatStore } from '../stores/imageChat.cjs'

const imageChatStore = useImageChatStore()
</script>

<template>
  <article class="grid grid-cols-1 lg:grid-cols-2 gap-x-8">
    <div class="mt-4">
      <h3>Add image and ask questions about it</h3>

      <section class="my-4">
        <input v-model="imageChatStore.imageURL" />
      </section>

      <img :src="imageChatStore.imageURL" class="w-80 mt-10" />
    </div>

    <div class="min-h-80 flex flex-col justify-between">
      <ChatWindowImage />

      <div class="flex justify-end">
        <button @click="imageChatStore.clearChat()" class="button button-secondary">Clear</button>
      </div>
    </div>
  </article>
</template>
